import cv2
import numpy as np
from PIL import Image

"""Test if webcam is working"""

vid = cv2.VideoCapture(0)
ret, frame = vid.read()


while True:
    ret, frame = vid.read()

    cv2.imshow("frame", frame)
    
    # Breakaway condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break