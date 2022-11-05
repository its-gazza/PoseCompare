from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torchvision.io import read_image
from torchvision.ops import nms
from torchvision.transforms import functional as F

from util.helpers import (
    connect_skeleton,
    get_angle,
    have_cuda,
    kpts_angle,
    model,
    transforms,
)



class PoseCompare:

    def __init__(self) -> None:
        None

    def inference(self, img) -> Tuple[Dict, Dict]:
        """Inference on an image

        Args:
                img (Tensor): Input tensor

        Returns:
                Tuple of output dictionary and angle dictionary

        Notes:
                Output dictionary: ``keypointrcnn_resnet50_fpn`` output
                Angle dictionary: See ``get_angle`` from helpers.py
        """
        # Read and transform image
        img = transforms(img)

        # Transfers to CUDA memory if available
        if have_cuda:
            img = img.cuda()

        # Inference
        output = model([img])[0]

        # Extract only the first person and convert all tensor to numpy
        # This will also release memory from CUDA
        for k, v in output.items():
            output[k] = v[0].cpu().detach().numpy()

        # Calculate angle
        angles = {}
        for k, v in kpts_angle.items():
            angles[k] = get_angle(output["keypoints"], v)

        return output, angles

    def draw_one(self, trgt: str = "ref", include_angles: bool = True) -> Image:
        """Draw reference or target image

        Args:
                trgt (str): Target image, acceptable input: ["trgt" or "ref"]
                include_angles (bool): Write joint angles to image. Default to True
        """
        # ==== Setup ==== #
        # ---- Variables ---- #
        # Assign values based on target or reference image
        if trgt == "ref":
            kp = self.output_ref["keypoints"]
            angle = self.angle_ref
            img = self.img_ref
            bbox = self.output_ref["boxes"]
        else:
            kp = self.output_trgt["keypoints"]
            angle = self.angle_trgt
            img = self.img_trgt
            bbox = self.output_trgt["boxes"]

        # Setup Font
        font = ImageFont.truetype("Arial.ttf", size=50)

        # ---- Image ---- #
        # Copy image
        img = img.copy()
        # Establish draw object
        img_draw = ImageDraw.Draw(img)

        # ==== Draw ==== #
        # Skeletons
        for con in connect_skeleton:
            # Get points in interest
            pt1, pt2 = con[0], con[1]

            # Determine start and end point to draw line
            start_x, start_y = kp[pt1][0], kp[pt1][1]
            end_x, end_y = kp[pt2][0], kp[pt2][1]

            # Draw line
            img_draw.line([(end_x, end_y), (start_x, start_y)], fill="red", width=3)

        # Facial Features
        for pt in range(3):
            pixel = 2
            pt1 = (kp[pt][0] - pixel, kp[pt][1] - pixel)
            pt2 = (kp[pt][0] + pixel, kp[pt][1] + pixel)

            # Draw eclipse: There's no circle method
            img_draw.ellipse([pt1, pt2], fill="red")

        # Angles
        if include_angles:
            angle_str = "\n".join([f"{k}: {v}" for k, v in angle.items()])

            img_draw.text((0, 0), angle_str, font=font)

        # Bounding boxes
        img_draw.rectangle([(bbox[0], bbox[1]), (bbox[2], bbox[3])])

        return img

    def draw_compare(self, fps: Optional[int] = None, offset: int = 20):
        """Draw comparison between reference and trgt

        Args:
                fps (int, Optional): Display FPS information. If set to None no FPS will be output. Default to None
                offset (int): Maximum Angle Diff. Default to 20
        """
        # ==== Setup ==== #
        draw = Image.new("RGB", (1200, 600))
        draw_ = ImageDraw.Draw(draw)
        font = ImageFont.truetype("Arial.ttf", size=20)

        # Draw inference
        draw_ref = self.draw_one(trgt="ref", include_angles=False)
        draw_trgt = self.draw_one(trgt="trgt", include_angles=False)

        # Image Box
        ref_box = draw_ref.crop(tuple(self.output_ref["boxes"])).resize((200, 400))
        trgt_box = draw_trgt.crop(tuple(self.output_trgt["boxes"])).resize((200, 400))

        # Angle text
        angle_ref = "\n".join([f"{k}: {v}" for k, v in self.angle_ref.items()])
        angle_trgt = "\n".join([f"{k}: {v}" for k, v in self.angle_trgt.items()])

        # ---- Maximum Angle Diff Calculation ---- #
        # Difference in angle
        angle_diff = self.calc_angle_diff()

        angle_diff_str = [f"Maximum Angle Diff: {offset}"]

        all_ok = True
        for k, v in angle_diff.items():
            if abs(v) < offset:
                status = f"OK ({v})"
            else:
                status = f"NOT OK ({v})"
                # If at least one is not OK then all_ok will be False
                all_ok = False

            angle_diff[k] = status

        angle_diff_str.extend([f"{k}: {v}" for k, v in angle_diff.items()])
        angle_diff_str.append(f"ALL OK: {all_ok}")
        angle_diff_str = "\n".join(angle_diff_str)

        # ==== Draw ==== #
        # Ref/Trgt to Image
        draw.paste(ref_box, (200, 150))
        draw.paste(trgt_box, (800, 150))

        # Straight Line in the middle
        draw_.line([(600, 0), (600, 600)])

        # Angles boxes
        draw_.text((10, 400), angle_ref, font=font)
        draw_.text((610, 400), angle_trgt, font=font)
        draw_.text((610, 0), angle_diff_str, font=font)

        # Who is who
        draw_.text((400, 0), "Reference", font=font)
        draw_.text((1000, 0), "Target", font=font)

        # FPS
        if fps is not None:
            draw_.text((1100, 580), f"FPS: {int(fps)}", font=font)

        return draw

    def load_img(self, frame, dest: str):
        """Inference trgt frame

        Args:
                frame: If a string is passed, it treats it as a path, else treat as a Tensor
                dest (str): Is the frame in question a reference or target? Acceptable input: ["ref", "trgt"]
        """
        # If string then read image and transforms image
        if type(frame) == str:
            # Read
            img = read_image(frame)
            img = transforms(img)
        else:
            # If array then convert to PIL
            img = Image.fromarray(frame)

        # Apply inference
        output, angles = self.inference(img)

        # Assign image and output to the relevant spot.
        if dest == "trgt":
            self.output_trgt = output
            self.angle_trgt = angles
            self.tensor_trgt = img
            self.img_trgt = F.to_pil_image(transforms(img))
        else:
            self.output_ref = output
            self.angle_ref = angles
            self.tensor_ref = img
            self.img_ref = F.to_pil_image(transforms(img))

    def calc_angle_diff(self) -> Dict[str, int]:
        """Calculate target and reference's angle difference

        Returns:
                A dictionary containing the joint as key, angle difference as the value
        """
        angle_diff = {k: self.angle_ref[k] - self.angle_trgt[k] for k in self.angle_ref}

        return angle_diff

