import cv2
from ultralytics import YOLO
import numpy as np
from paddleocr import PaddleOCR
import requests
from dotenv import load_dotenv
import os
import time
import json

load_dotenv()
DEBUG = os.getenv('DEBUG') == 'True'
PREVIEW = os.getenv('PREVIEW') == 'True'

def debug(message):
    if DEBUG:
        print(message)

# Checkpoint class
class Checkpoint:
    def __init__(self):
        self.cameras = []
        self.uuid = None
        self.username = os.getenv('API_USERNAME')
        self.password = os.getenv('API_PASSWORD')
        self.api_url = os.getenv('API_URL')
        self.set_cameras()

    def set_cameras(self):
        url = f"{self.api_url}/checkpoint/cameras"

        try:
            response = requests.get(url, auth=(self.username, self.password))
            response.raise_for_status()
            config = response.json()

            # Initialize cameras
            self.cameras = [
                Camera(
                    uuid=camera["uuid"],
                    username=camera["username"],
                    password=camera["password"],
                    ip=camera["ip"],
                    port=camera["port"],
                    url=camera["url"],
                    recognition=camera["recognition"]
                )
                for camera in config.get("cameras", [])
            ]

            debug(f"Number of cameras: {len(self.cameras)}")
        except requests.exceptions.RequestException as e:
            debug(f"Error fetching checkpoint data: {e}")
    
    # Request images from server wgicj should captured from cameras
    def get_images(self):
        url = f"{self.api_url}/image/search"

        try:
            response = requests.get(url, auth=(self.username, self.password))
            response.raise_for_status()
            data = response.json()

            # Extract images from the response
            images = data.get("images", [])
            
            if not images:
                return None

            # Filter images based on checkpoint cameras
            camera_uuids = {camera.uuid for camera in self.cameras}
            filtered_images = [
                image for image in images if image["camera_uuid"] in camera_uuids
            ]
            
            return filtered_images
        except requests.exceptions.RequestException as e:
            return None

    # This method retrieves the camera object associated with a given image.    
    def get_image_camera(self, image):
        for camera in self.cameras:
            if camera.uuid == image["camera_uuid"]:
                return camera

        return None
        

# Camera class
class Camera:
    def __init__(self, uuid, username, password, ip, port, url, recognition):
        self.uuid = uuid
        self.username = self.decode_credentials(username)
        self.password = self.decode_credentials(password)
        self.ip = ip
        self.port = port
        self.url = self.get_url(url)
        self.recognition = 'xxx' #recognition
        self.capture = None

    # Decode the credentials
    def decode_credentials(self, credentials):
        salt_length = int(os.getenv('SALT_LENGTH', 0))
        
        if salt_length <= 0:
            return None

        # Decode the string by selecting characters at indices proportional to SALT_LENGTH + 1
        decoded = ''.join([
            char 
            for i, char in enumerate(credentials) 
            if (i % (salt_length + 1)) == 0
        ])
        return decoded

    
    # Returns the URL for the camera stream by replacing placeholders with actual values.
    # The placeholders are [username], [password], [ip], and [port].
    def get_url(self, url_template):
        values = {
            'username': self.username,
            'password': self.password,
            'ip': self.ip,
            'port': self.port,
        }
        url = url_template

        for key, value in values.items():
            url = url.replace(f"[{key}]", str(value))

        return url

    def get_frame(self):
        capture = cv2.VideoCapture(self.url)

        # Capture a frame from the camera
        ret, frame = capture.read()
        capture.release()

        if not ret:
            return None

        return frame

# Anpr class
# This class is responsible for performing ANPR (Automatic Number Plate Recognition) using Yolo and PaddleOCR.
class Anpr():
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.detector = YOLO('anpr.pt')
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en')

    def perform_ocr(self, roi):
        if not isinstance(roi, np.ndarray):
            return None

        results = self.ocr.ocr(roi, rec=True)
 
        if results and results[0]:
            extracted_text = [result[1][0] for result in results[0]]
            return ' '.join(extracted_text).strip()

        return ''

    def detect_plate(self, frame):
        results = self.detector(frame)

        plate = None
        plate_box = None
        max_confidence = 0.0

        for result in results:
            boxes = result.boxes.xyxy  # [x1, y1, x2, y2]
            confidences = result.boxes.conf  # confidence

            for box, confidence in zip(boxes, confidences):
                if (plate_box is None) or (confidence > max_confidence):
                    plate_box = box
                    max_confidence = confidence

        # Extract the plate region
        if plate_box is not None:
            x1, y1, x2, y2 = map(int, plate_box)
            plate = np.array(frame)[y1:y2, x1:x2]

        return (plate, plate_box)

# Initialize the object counter
anpr = Anpr()
checkpoint = Checkpoint()
camera_unavailable = False

while True:
    time.sleep(5)
    # Try to fetch cameras if not available yet
    if not checkpoint.cameras or camera_unavailable:
        checkpoint.set_cameras()
        camera_unavailable = False
        debug("No cameras available")
        time.sleep(60)
        continue

    images = checkpoint.get_images()

    # Check if images are available
    if not images:
        debug("No images available")
        continue

    payload = []

    for image in images:
        camera = checkpoint.get_image_camera(image)

        # Fail image processing if camera is not found
        if camera is None:
            payload.append({
                'uuid': image['uuid'],
                'status': 'failed',
            })
            continue

        # checkpoint.cameras
        try:
            # Read the frame from the camera
            frame = camera.get_frame()

            # Check if camera was able to fetch the frame, if not try to reinitialize
            if frame is None:
                camera_unavailable = True
                break

            frame = cv2.resize(frame, (1024, 768))

            recognition_text = None
            plate_roi = None
            plate_box = None

            # Perform ANPR processing
            if camera.recognition == 'licence_plate':
                # Detect license plate
                (plate_roi, plate_box)  = anpr.detect_plate(frame)
                # Recognize license plate number
                recognition_text = anpr.perform_ocr(plate_roi)

            payload.append({
                'uuid': image['uuid'],
                'status': 'processed',
                'recognition_text': recognition_text,
                'image': cv2.imencode('.jpg', frame)[1].tobytes()
            })
            
            if PREVIEW:
                # Display the frame with the recognized text
                if recognition_text:
                    cv2.putText(frame, recognition_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if plate_box is not None:
                    x1, y1, x2, y2 = map(int, plate_box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Show the frame in a window
                cv2.imshow(camera.uuid, frame)
                cv2.waitKey(1)

        except ValueError as e:
            payload.append({
                'uuid': image['uuid'],
                'status': 'failed',
            })
            debug(e)
    
    # Send the results to the server
    url = f"{checkpoint.api_url}/image/upload"

    # Prepare the multipart form data
    files = {}
    data = {}
    i = 0

    for item in payload:
        # Extract the binary image data
        image_data = item.pop('image', None)
        
        if image_data:
            # Add the binary image to the files list
            files[f"images[{i}][image]"] = (f"{item['uuid']}.jpg", image_data, 'image/jpeg')
        
        for key, value in item.items():
            if value is not None:
                data[f"images[{i}][{key}]"] = item[key]

        i += 1

    # Send the request with multipart form data
    try:
        debug("Sending data")
        response = requests.post(
            url,
            data=data,  # Send the list directly as JSON
            files=files,  # Attach the binary images
            auth=(checkpoint.username, checkpoint.password)
        )
        response.raise_for_status()

        debug(response.json())
    except requests.exceptions.RequestException as e:
        debug(f"Error sending data: {e}")