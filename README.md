# Get Poses from Any Dynamic Videos
This repo provides streamlined scripts to obtain 6D camera poses from videos, even the ones with dynamic objects. The method comes from [EPIC-FIELDS](https://epic-kitchens.github.io/epic-fields/)。

## Installation
Follow instructions in the EPIC-FIELDS [repo](https://github.com/epic-kitchens/epic-Fields-code).

## Usage
### Get poses from any video
`get_pose_any_videos.py` is the main script used to get poses from any videos. It entails 3 stages:
1. Select sparse frames from homography
2. Reconstruct sparse frames
3. Register dense frames

It assumes a required argument `input_video_frame_paths` which directory contains a set of images (jpg or png). The output will be at the `colmap_models` folder with COLMAP output format. Additional information can be found at `logs`, `sampled_frames`.

### (Optional) Check pose accuracy with dycheck dataset GT
Use `check_pose_with_gt_dycheck.py`. Do remember to change some paths including `DATA_ROOT`, `SEQUENCE` and line 143.