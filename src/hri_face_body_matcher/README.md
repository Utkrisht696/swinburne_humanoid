# hri_face_body_matcher

## Overview

`hri_face_body_matcher` is a [ROS4HRI](https://wiki.ros.org/hri)-compatible face
to body matcher node.

It finds the most likely matches between the recognized faces and bodies based
on their relative position in the source image.

### Algorithm

For each of the possible associations of recognized face and body, a matching
cost is computed, linearly decreasing with the distance between the body nose
and the face center in the image.

The rate the confidence drops is proportional to the
`~confidence_scaling_factor` parameter and the face size, intended as its
diagonal length.
```math
confidence = max(0, 1 - \frac{distance * c.s.f.}{2 * face\ size})
```


## ROS API

### Parameters

All parameters are loaded in the lifecycle `configuration` transition.

- `~confidence_threshold` (double ∈ [0,1], default: 0.5):
  Candidate matches with confidence lower that this threshold are not published.

- `~confidence_scaling_factor` (double > 0, default: 2.0):
  Factor scaling how quickly the estimated confidence drops as the distance between the matched face and body increases.

### Topics

This package follows the ROS4HRI conventions ([REP-155](https://www.ros.org/reps/rep-0155.html)).
If the topic message type is not indicated, the ROS4HRI convention is implied.

#### Subscribed

- `/humans/bodies/tracked`
- `/humans/bodies/<body_id>/skeleton2d`
- `/humans/faces/tracked`
- `/humans/faces/<face_id>/roi`

#### Published

- `/humans/candidate_matches`

## Execution

```bash
ros2 launch hri_face_body_matcher hri_face_body_matcher.launch.py
```

## Example

For an example of usage, execute in different terminals:
- USB camera:
  1. `apt install ros-humble-usb-cam`
  2. `ros2 run usb_cam usb_cam_node_exe`
- HRI face detect:
  1. Either
    - if you are on a PAL robot `apt install ros-humble-hri-face-detect`
    - otherwise build and install from [source](https://github.com/ros4hri/hri_face_detect).
  2. `ros2 launch hri_face_detect face_detect.launch.py`
- HRI fullbody:
  1. Either
    - if you are on a PAL robot `apt install ros-humble-hri-fullbody`
    - otherwise build and install from [source](https://github.com/ros4hri/hri_face_fullbody).
  2. `ros2 launch hri_fachri_fullbody hri_fullbody.launch.py`
- HRI face body matcher:
  1. `apt install ros-humble-hri-face-body-matcher`
  2. `ros2 launch hri_face_body_matcher hri_face_body_matcher.launch.py`
- RViz with HRI plugin:
  1. `apt install ros-humble-rviz2`
  2. Either
    - if you are on a PAL robot `apt install ros-humble-hri-rviz`
    - otherwise build and install from [source](https://github.com/ros4hri/hri_rviz).
  3. `rviz2`

In RViz, add the 'Humans' plugin to see the detected faces.
The face IDs should be permanently assigned to the same people.
