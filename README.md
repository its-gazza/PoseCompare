# PoseCompare

The goal of `PoseCompare` is to compare two different pose at the joint level and provide actioable insight to users to
help them improve their posture/pose.

**NOTES:**

* Ensure webcam id is valid. You can test this using the `util/webcam_test.py` and alter line 7: (e.g.
  `cv2.VideoCapture(0)` instead of `cv2.VideoCapture(1)`)
* It is strongly recommended to run this with a GPU. Running with CPU will result in extreme slowness (~0.2FPS on an
  Intel i7 CPU). For comparison, running this on a RTX2080 8GB will yield 6FPS

## Inference

`PoseCompare` can inference on single image alone:

![Inference](./img/inference.png)

Or it can be used in comparing with another image:

![Compare](./img/compare.png)

## Install

To insatll, run:

```shell
    pip3 install -r requirements.txt
```

## Usage

A simple usage is as follow:

```python
>>> from util.pose_compare import PoseCompare
>>> # Initialize
>>> pose = PoseCompare()
>>> 
>>> # Inference Image
>>> pose.load_img(frame="data/tree.jpeg", dest="ref")
>>>
>>> # Draw Image
>>> pose.draw_one(trgt="ref")
```

## Contents of this repo

| File Name | Description |
| - | - |
| `data/*.jpeg` <br> `data/*.mp4` | Image/Video files used in example scripts |
| `util/helper.py` | Helper file for `util/pose_compare.py`, contains keypoints mapping, joint keypoints mapping etc. |
| `util/pose_compare.py` | Contains the `PoseCompare` class that is used to do pose estimation and pose comparison |
| `Arial.ttf` <br> `util/Arial.ttf` | Arial font, this is needed as Windows does not have this font in Library |
| `img_inference.ipynb` | Apply inference on a single image |
| `ref_vid_inference.py` | Apply inference on a video file |
| `webcam_inference.py` | Apply inference from webcam image |
| `img_webcam_compare.py` | Inference webcam image (Target) and compare with still image (Reference) |
