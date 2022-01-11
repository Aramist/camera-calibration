import argparse
import os
from os import path

import cv2
import numpy as np


def get_arguments():
    ap = argparse.ArgumentParser(description='Computes the intrinsic parameters of a camera from images of a checkerboard pattern.')
    ap.add_argument('image_dir', type=str, help='The directory containing the calibration images')
    ap.add_argument('output_dir', type=str, help='Where the intrinsic matrices should be written')
    ap.add_argument('--board_width', type=int, help='The width of the checkerboard (in tiles)', default=8)
    ap.add_argument('--board_height', type=int, help='The height of the checkerboard (in tiles)', default=6)
    return ap.parse_args()


def get_board_coordinates(board_shape):
    """ Returns an array of shape (1, width * height, 3) where each point (dim -1) represents the
    location of a corner in the checkerboard space. In this reference frame, the origin is placed
    at the top-left point.
    The points in this array are ordered in a row major (0-dim) fashion:
    [[ (0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0), ..., (0, 1, 0), (1, 1, 0), ... ]]
    Originally this was done with np.mgrid but I opted for this less-efficient implementation for
    the sake of readability
    Args:
      - board_shape: tuple (2,)
        A tuple of the form (rows, cols), where rows and cols are ints
    """

    rows, cols = board_shape
    return_array = np.zeros((1, rows * cols, 3), dtype=np.float32)

    # For each index (0, n), the point will be of the form (n % rows, n // rows, 0)
    # Thus, the x coordinate is always less than width and the y coordinate ultimately reaches cols - 1
    index = np.arange(rows * cols)
    x = index % rows
    y = index // rows
    # Leave z alone because it's already zero
    return_array[0, :, :2] = np.stack([x, y], axis=1)
    return return_array


def sample_image_shape(image_paths):
    """ Helper function to get the shape of the training images
    This is necessary because knowledge of the image shape is a prerequisite for computing
    the intrinsics and it's much cleaner to calculate this outside of the other calibration
    code and avoid global variables altogether.
    Assumes the training image folder is not empty
    """
    im = cv2.imread(image_paths[0])
    return im.shape[:2]


def detect_board_corners(img, board_size, corners_criteria, subpixel_criteria):
    """ Computes the locations of the checkerboard corners to sub-pixel accuracy and returns them as a list
    Args:
        img: ndarray (height, width, 3 or 1)
          - Image of the checkerboard. Can be grayscale or bgr
        board_size: tuple (width, height)
          - Unit width and height of the checkerboard pattern in the images
        corners_criteria: bitmask (int?)
          - A bitmask of opencv constants determining the options for the corner detection algorithm
        subpixel_criteria: bitmask (int?)
          - Similar to corners_criteria, but specific to the sub-pixel algorithm
    Return: a list of ndarray, each representing a point on the board corresponding to the real world
    point at the same index if the operation succeeded. None if the operation failed.
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    ret, rough_corners = cv2.findChessboardCorners(
            gray,
            board_size,
            corners_criteria
    )

    if ret:
        corners_refined = cv2.cornerSubPix(
                gray,
                rough_corners,
                (5, 5),
                (-1, -1),
                subpixel_criteria
        )
        return corners_refined
    return None


def run():
    """ Performs the camera calibration using the fisheye module of opencv
    The process can be roughly divided into three stages:
      1) Locate the corners (points surrounded by four board tiles) of the checkerboard with sub-pixel accuracy
      2) Regress the camera parameters through the mapping from camera coordinates (the board corner locations)
         to real world coordinates (a coordinate frame with its origin placed at the top-left corner of the board
         with one unit being the width of a tile).
      3) (Outside this script) Use the camera's intrinsic parameters to compute a mapping for the inverse
         transformation and apply it to the image, cropping out any distorted corners resulting from the operation.

    Note that all images must have the same shape (which should already be the case, if they're taken by the same
    camera)
    """

    # Setup: get arguments, define input/output directories, and define constants for the calibration functions
    args = get_arguments()
    calibration_image_dir = args.image_dir
    output_param_dir = args.output_dir

    if not path.exists(calibration_image_dir):
        raise Exception('Failed to locate the given calibration image directory: {}'.format(calibration_image_dir))
    if path.exists(path.join(output_param_dir, 'K.npy')):
        os.remove(path.join(output_param_dir, 'K.npy'))
    if path.exists(path.join(output_param_dir, 'D.npy')):
        os.remove(path.join(output_param_dir, 'D.npy'))
    if not path.exists(output_param_dir):
        os.mkdir(output_param_dir)

    # Filter out any non-image files
    calibration_images = [
            path.join(calibration_image_dir, fname)
            for fname in os.listdir(calibration_image_dir)
            if fname.endswith('.png') or fname.endswith('.jpg')
    ]

    if not calibration_images:
        raise Exception('Failed to locate any calibration images in the directory {}'.format(calibration_image_dir))

    # OpenCV is very inconsistent about order when heights and widths are involved
    # Here it wants the shape as num_rows, num_cols
    board_size = (args.board_height, args.board_width)
    board_points_criteria = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    board_points_subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    # The CHECK_COND flag sometimes has problems with specific training images and it's difficult to
    # determine which image was problematic unless you label your image files and insert them in sorted
    # order. I don't think including the problematic images negatively impacts the regression so i'm
    # excluding that flag
    calibration_flags = (
            cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC \
            + cv2.fisheye.CALIB_FIX_SKEW \
            + cv2.fisheye.CALIB_CHECK_COND
    )
    # I think I had this set to 1e-3 in the past and the results were terrible
    calibration_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-6)


    world_points = list()
    image_points = list()
    world_coord_frame = get_board_coordinates(board_size)
    image_shape = sample_image_shape(calibration_images)

    # For each image, find the board corners. If they exist, append a (world points, image points) pair for the image
    for image in calibration_images:
        cv_img = cv2.imread(image)
        corners = detect_board_corners(
                cv_img,
                board_size,
                board_points_criteria,
                board_points_subpix_criteria
        )

        if corners is not None:
            world_points.append(world_coord_frame)
            image_points.append(corners)

    # Now run the calibration algorithm on the pairs:
    # Placeholders for all of the algorithm's variables
    num_samples = len(world_points)
    if num_samples == 0:
        raise Exception('Could not locate chessboard corners. Perhaps the board size wasn\'t set properly?')
    k_matrix = np.zeros((3, 3))
    d_matrix = np.zeros((4, 1))
    r_vecs = [np.zeros((1, 1, 3), dtype=float) for _ in range(num_samples)]
    t_vecs = [np.zeros((1, 1, 3), dtype=float) for _ in range(num_samples)]

    cv2.fisheye.calibrate(
        world_points,
        image_points,
        image_shape[::-1],  # Reverse the image shape to turn (height, width) into (width, height)
        k_matrix,
        d_matrix,
        r_vecs,
        t_vecs,
        calibration_flags,
        calibration_criteria
    )

    # Finally, save the intrinsic parameters:
    np.save(path.join(output_param_dir, 'K.npy'), k_matrix)
    np.save(path.join(output_param_dir, 'D.npy'), d_matrix)
    print('K: {}'.format(str(k_matrix)))
    print('D: {}'.format(str(d_matrix)))


if __name__ == '__main__':
    run()

