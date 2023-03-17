import cv2
import numpy as np

input_video_path = './Shot-Predictor-Video.mp4'

cap = cv2.VideoCapture(input_video_path)

import cv2
import numpy as np

# Create a named window for the trackbars
cv2.namedWindow('Trackbars')

# Create trackbars for HSV thresholds
cv2.createTrackbar('Hue Low', 'Trackbars', 0, 179, lambda x: None)
cv2.createTrackbar('Hue High', 'Trackbars', 179, 179, lambda x: None)
cv2.createTrackbar('Sat Low', 'Trackbars', 0, 255, lambda x: None)
cv2.createTrackbar('Sat High', 'Trackbars', 255, 255, lambda x: None)
cv2.createTrackbar('Val Low', 'Trackbars', 0, 255, lambda x: None)
cv2.createTrackbar('Val High', 'Trackbars', 255, 255, lambda x: None)

# Loop over frames
while True:
    # Read a frame from the video
    ret, frame = cap.read()
    
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Get the current trackbar values
    h_low = cv2.getTrackbarPos('Hue Low', 'Trackbars')
    h_high = cv2.getTrackbarPos('Hue High', 'Trackbars')
    s_low = cv2.getTrackbarPos('Sat Low', 'Trackbars')
    s_high = cv2.getTrackbarPos('Sat High', 'Trackbars')
    v_low = cv2.getTrackbarPos('Val Low', 'Trackbars')
    v_high = cv2.getTrackbarPos('Val High', 'Trackbars')
    cv2.setTrackbarPos('Hue Low', 'Trackbars', h_low)
    cv2.setTrackbarPos('Hue High', 'Trackbars', h_high)
    cv2.setTrackbarPos('Sat Low', 'Trackbars', s_low)
    cv2.setTrackbarPos('Sat High', 'Trackbars', s_high)
    cv2.setTrackbarPos('Val Low', 'Trackbars', v_low)
    cv2.setTrackbarPos('Val High', 'Trackbars', v_high)
    h_low = cv2.getTrackbarPos('Hue Low', 'Trackbars')
    h_high = cv2.getTrackbarPos('Hue High', 'Trackbars')
    s_low = cv2.getTrackbarPos('Sat Low', 'Trackbars')
    s_high = cv2.getTrackbarPos('Sat High', 'Trackbars')
    v_low = cv2.getTrackbarPos('Val Low', 'Trackbars')
    v_high = cv2.getTrackbarPos('Val High', 'Trackbars')
    # Threshold the image based on the current trackbar values
    lower = np.array([h_low, s_low, v_low])
    upper = np.array([h_high, s_high, v_high])
    mask = cv2.inRange(hsv, lower, upper)
    # Apply the mask to the original image
    result = cv2.bitwise_and(frame, frame, mask=mask)
    # Display the resulting image
    cv2.imshow('Result', result)
    # Wait for a key press and update the trackbar values
    cv2.waitKey(300)
cv2.destroyAllWindows()
cap.release()



