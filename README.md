# Fisheye calibration

Removes fisheye distortion.

## Description

Contains two scripts, calibrate.py and undistort.py, which respectively handle calibration of the camera over a set of training images and post-calibration transformations of any images or videos captured by the same camera to remove fisheye distortion.

## Getting Started

### Dependencies

* Python 3, OpenCV 3, NumPy
```
python3 -m pip install opencv-python, numpy
```

### Installing

* Ensure all prerequisites are installed
* Clone the repository:
```
git clone https://github.com/Aramist/camera-calibration
```

### Executing program

* Print a checkerboard pattern onto a sheet of paper and affix it to a flat surface. Take note of its width and height.
* Produce a set of calibration images, 20 is more than sufficient. Try to capture the checkerboard pattern from multiple angles and positions
* Move the calibration images into a single folder, calibration\_images in the sample execution below, and run calibrate.py with the board\_width and board\_height parameters equal to the width and height (in tiles) of the board **minus 1**<sup>[1]</sup>
* To verify the calibration, try running undistort.py over the entire calibration image directory (line 3 of the sample execution below). If the results look strange, double-check the order of the board size parameters and consider removing some training images taken from ambiguous perspectives.
* Once you have a good calibration, the undistort.py script can be used to transform individual images and videos or entire directories of images and videos.
Sample execution:
```
python3 calibrate.py --board_width 8 --board_height 6 calibration_images camera_params
python3 undistort.py camera_params distorted_video.avi undistorted_video.avi
python3 undistort.py camera_params calibration_images undistorted_media
```

## Notes

Only .avi files can be written

## Authors

Aramis Tanelus

## Version History

* 0.1
    * Initial Release

## Acknowledgments

Inspiration, code snippets, etc.
* [This Medium article on the topic](https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-333b05afa0b0)

[1] The board\_width and board\_height don't actually refer to the tiles, but to the matrix of points formed by the intersections of the lines running horizontally and vertically between the tiles. This does not include the points along the edge of the board, so the width and height of this matrix ends up being that of the board minus one in each dimension.
