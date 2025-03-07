import cv2
import face_recognition
import os
import numpy as np
import time

# Directory for storing registered user faces
KNOWN_FACES_DIR = "known_faces"
known_face_encodings = []
known_face_names = []

# Load and encode registered user faces
for filename in os.listdir(KNOWN_FACES_DIR):
    image_path = os.path.join(KNOWN_FACES_DIR, filename)
    image = face_recognition.load_image_file(image_path)
    encoding = face_recognition.face_encodings(image)[0]
    
    known_face_encodings.append(encoding)
    known_face_names.append(os.path.splitext(filename)[0])  # Remove file extension

# Initialize webcam
video_capture = cv2.VideoCapture(0)

print("Press 'q' to exit")

# Variables for liveness detection
blink_start_time = None
eye_closed_frames = 0
BLINK_THRESHOLD = 3  # Number of frames eyes must be closed for a blink

while True:
    ret, frame = video_capture.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            match_index = matches.index(True)
            name = known_face_names[match_index]

        # Liveness detection using eye blinking
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eyes = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
        detected_eyes = eyes.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(detected_eyes) == 0:  # If eyes are closed
            eye_closed_frames += 1
            if eye_closed_frames >= BLINK_THRESHOLD:
                print("Blink detected! Liveness confirmed.")
                eye_closed_frames = 0  # Reset blink counter
        else:
            eye_closed_frames = 0  # Reset if eyes are open

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Secure Face Authentication", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
video_capture.release()
cv2.destroyAllWindows()
