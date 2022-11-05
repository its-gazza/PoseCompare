import math
from typing import Dict, List, Tuple

import PIL
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.models.detection import (
    KeypointRCNN_ResNet50_FPN_Weights,
    keypointrcnn_resnet50_fpn,
)
from torchvision.transforms import functional as F

"""PoseCompare Helper file

This script contains helper functions and models that is used
"""

# ==== Model ==== #
weights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
transforms = weights.transforms()

model = keypointrcnn_resnet50_fpn(weights=weights, progress=True)
model = model.eval()

# ---- Enable CUDA if available---- #
# Determine CUDA
have_cuda = torch.cuda.is_available()
# Enable CUDA for model
if have_cuda:
    device = torch.device("cuda")
    model.to(device)
    print("MODEL: CUDA Enabled")
else:
    print("MODEL: CUDA NOT Enabled")

# ==== Variable ==== #
# Skeleton that we're interested in
connect_skeleton = [
    (5, 7),  # Left shoulder to elbow
    (6, 8),  # Right shoulder to elbow
    (11, 12),  # Left and right hip
    (5, 6),  # Left and right shoulder
    (7, 9),  # Left elbow to wrist
    (8, 10),  # Right elbow to wrist
    (5, 11),  # Left shoulder and hip
    (6, 12),  # Right shoulder to hip
    (11, 13),  # Left hip to knee
    (12, 14),  # Right hip to knee
    (13, 15),  # Left knee to ankle
    (14, 16),  # Right knee to ankle
]

# Keypoints mapping
kpts_name = {
    "0": "nose",
    "1": "left_eye",
    "2": "right_eye",
    "3": "left_ear",
    "4": "right_ear",
    "5": "left_shoulder",
    "6": "right_shoulder",
    "7": "left_elbow",
    "8": "right_elbow",
    "9": "left_wrist",
    "10": "right_wrist",
    "11": "left_hip",
    "12": "right_hip",
    "13": "left_knee",
    "14": "right_knee",
    "15": "left_ankle",
    "16": "right_ankle",
}

# keypoints id required to calculate angle
kpts_angle = {
    "right_shoulder": [5, 6, 8],
    "right_arm": [6, 8, 10],
    "left_shoulder": [7, 5, 6],
    "left_arm": [9, 7, 5],
    "right_hip": [11, 12, 14],
    "right_leg": [12, 14, 16],
    "left_hip": [13, 11, 12],
    "left_leg": [11, 13, 15],
}

# ==== Functions ==== #
def get_angle(kpts_coord: List[Tuple[int, int]], angle_kpts: List[Tuple[int, int]]):
    """Get angle from 3 keypoints

    This function will calculate the angle between 3 points.

    Args:
        kpts_coord: keypoints cooradiate, should be 17 length long
        angle_kpts: List of keypoints to get angle from. It should be 3 length
            long with the angle point in the middle
    """
    # Get coords from the 3 points
    a = kpts_coord[angle_kpts[0]]
    b = kpts_coord[angle_kpts[1]]
    c = kpts_coord[angle_kpts[2]]

    # Calculate angle
    ang = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    )

    # Sanity check, angle must be between 0-180
    ang = int(ang + 360 if ang < 0 else ang)
    ang = int(ang - 180 if ang > 270 else ang)

    return ang


def draw_kp(
    img: torch.Tensor,
    output: Dict,
    connect_skeleton: List[Tuple[int, int]],
    background: str = "image",
) -> PIL.Image.Image:
    """Draw keypoints

    Args:
        img (torch.Tensor): _description_
        output (Dict): _description_
        connect_skeleton (List[Tuple[int, int]]): _description_
        background (str, optional): Backgeround image, possible values ["image", "blank"]. Defaults to "image".

    Returns:
        PIL.Image.Image: _description_
    """

    # Extract keypoints co-ordinates
    kp = output["keypoints"][0]

    # ==== Image setup ==== #
    if background == "image":
        draw = F.to_pil_image(img)
    else:
        draw = Image.new("RGB", img.numpy().shape[1:][::-1])  # Back background

    # ImageDraw setup
    draw1 = ImageDraw.Draw(draw)
    font = ImageFont.load_default()

    # ==== Joint Angles ==== #
    # Calculate angles
    angles = []
    for k, v in kpts_angle.items():
        angles.append(f"{k}: {get_angle(kp, v)}")

    # ==== Draw ==== #
    # Connection Skeleton
    for con in connect_skeleton:
        pt1 = con[0]
        pt2 = con[1]
        start_x, start_y = kp[pt1][0], kp[pt1][1]
        end_x, end_y = kp[pt2][0], kp[pt2][1]
        draw1.line([(end_x, end_y), (start_x, start_y)], fill=128, width=10)

    # Facial keypoints for aesthetics
    for pt in range(3):
        pixel = 2
        pt1 = (kp[pt][0] - pixel, kp[pt][1] - pixel)
        pt2 = (kp[pt][0] + pixel, kp[pt][1] + pixel)

        draw1.ellipse([pt1, pt2], fill="red")

    angles = "\n".join(angles)

    # Draw angles
    draw1.text((0, 0), angles, font=font)

    return draw
