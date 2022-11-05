import cv2
import numpy as np
from util.pose_compare import PoseCompare


if __name__ == "__main__":
    # ==== Setup === #

    # ---- Webcam ---- #
    # Setup cv2 class
    vid = cv2.VideoCapture(0)
    # Read first image for sanity check
    _, frame = vid.read()

    # ---- PoseCompare ---- #
    # Initialzie
    pose = PoseCompare()

    # ==== Main Loop ==== #
    while True:
        # Read frame
        ret, frame = vid.read()

        # Exit if end of video
        if not ret:
            break

        # ==== PoseCompare ==== #
        # Load frame and inference
        pose.load_img(frame=frame, dest="ref")

        # Draw image and output as np.array to allow cv2 to show
        output = pose.draw_one(trgt="ref")
        output = np.array(output)

        # ==== Show Video ==== #
        cv2.imshow("Inference", output)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
