import cv2
import tempfile
import os
import numpy as np
from datetime import datetime
from inference_sdk import InferenceConfiguration, InferenceHTTPClient
import pygame
import time

# Initialize Roboflow model
model_id = "challenge-3-laminotomy-omi7k/2"
roboflow_api_key= "eobEFVMXPwTYv004igpH"
camera_capture_port = 0

confidence_threshold = 0.5
iou_threshold = 0.5

config = InferenceConfiguration(confidence_threshold, iou_threshold)

# Initialize pygame
pygame.init()

print(config.confidence_threshold)
client = InferenceHTTPClient(
    api_url="http://localhost:9001",
    api_key=roboflow_api_key
)
client.configure(config)
client.select_model(model_id)

class_ids = {}

# Toggle to turn on AI mode
ai_mode = 0

# Start webcam
cap = cv2.VideoCapture(camera_capture_port)

# Get the width and height of the frame from the capture device
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create a VideoWriter object to save the video
# The filename includes the current date and time
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
out_filename = f"record_{current_time}.mp4"
out = cv2.VideoWriter(out_filename, cv2.VideoWriter_fourcc(*'mp4v'), 7.0, (frame_width, frame_height))

ai_mode = 0
exit = 0

# Initialize FPS variables
fps = 0
frame_count = 0
start_time = time.time()

while not exit: 
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame_count += 1
    current_time = time.time()
    
    # Calculate FPS every second
    if current_time - start_time >= 1:
        fps = frame_count
        frame_count = 0
        start_time = current_time

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Save the frame temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        cv2.imwrite(tmp_file.name, cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
        # Now tmp_file.name contains the path to the saved image
        
        if ai_mode:
            prediction = client.infer(tmp_file.name)
            predictions_list = prediction['predictions']  # Access the 'predictions' key directly
            for detection in predictions_list:
                points = detection['points']
                contour = np.array([[int(point['x']), int(point['y'])] for point in points], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [contour], isClosed=True, color=(0, 255, 0), thickness=2)

        # Ensure to delete the temp file after using it
        os.unlink(tmp_file.name)

    # Display FPS on the frame
    cv2.putText(frame, f'FPS: {fps}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Write the frame into the file 'out'
    out.write(frame)
    
    # Display the resulting frame
    cv2.imshow('Frame', frame)
    
    pygame.event.pump()

    for event in pygame.event.get():    
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:                
                exit = 1 
            elif event.key == pygame.K_a:
                if ai_mode:
                    ai_mode = 0
                    print("AI Deactivated")
                else:
                    ai_mode = 1
                    print("AI Activated")

# Release the capture
cap.release()
cv2.destroyAllWindows()
out.release()
