import cv2

# Load the cascade
face_identifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# To capture video from webcam.
capture = cv2.VideoCapture(0)
# To use a video file as input
# capture = cv2.VideoCapture('filename.mp4')

while True:
    # Read the frame
    _, img = capture.read()
    # Convert to grayscale
    gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = face_identifier.detectMultiScale(gray_scale, 1.5, 6)
    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # Display
    cv2.imshow('img', img)
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
# Release the VideoCapture object
capture.release()