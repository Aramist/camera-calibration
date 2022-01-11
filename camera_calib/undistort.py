import argparse
import os
from os import path
import re
import time

import cv2
import numpy as np


""" Caveats:
    1) The ratio parameter is constant, where it would make sense for it to be uniquely determined for each
    input. This can be solved by requiring the dimensions of the training image as an input to the program, but
    I avoided that because it seemed unintuivite. At the same time, keeping a second config file seemed strange
    for a script as small as this.
    2) Here, the program fails for non-avi input videos, but in reality these can be read. The only issue is
    that non-avi videos cannot be written so the file name would have to change.
    3) Doesn't filter out files without extensions / files without known extensions, so things like .DS_STORE
    will trigger error messages that don't impact the functionality of the script
"""


IMAGE_FILE_EXTENSIONS = ('png', 'jpg', 'jpeg')


def get_arguments():
    ap = argparse.ArgumentParser('A script for undistorting images based on provided camera intrinsics')
    ap.add_argument('parameter_path', type=str, help='Directory containing K.npy and D.npy')
    ap.add_argument('input_path', type=str, help='A path to the input directory (in the case of image batch undistortion) or input file (for single images or videos).')
    ap.add_argument('output_path', type=str, help='A path to the output directory or file')
    ap.add_argument('--ratio', type=float, help='The ratio between the training image width or height and the images being undistorted.', default=1)
    return ap.parse_args()


def compute_maps(frame, K, D, ratio=1):
    dims = frame.shape[:2][::-1]
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, dims, np.eye(3), balance=0)
    new_K *= ratio
    new_K[2, 2] = 1  # The bottom right element of the matrix should always be 1
    maps = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, dims, cv2.CV_16SC2)
    return maps


def undistort_frame(frame, maps=None, K=None, D=None, ratio=1):
    """ Given either a set of precalculated maps or a K/D pair, undistorts an image
    Args:
        frame: ndarray
          - The image to undistort
        maps: tuple (2,)
          - A set of maps for the undistortion transformation
        K: ndarray (3, 3)
          - The camera's intrinsic parameter matrix
        D: ndarray (4, 1)?
          - Distortion parameters for the camera (I don't actually know what they mean)
        ratio: float
          - The ratio of the training image width to the given image width. Useful for
          training on full images and undistorting downsized images. Aspect ratio must
          be constant between the training images and the applied images.
    """
    if maps is not None:
        return cv2.remap(frame, *maps, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    if K is None or D is None:
        raise Exception("No maps or intrinsic parameters were given to perform the undistortion transformation")
    dims = frame.shape[:2][::-1]
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, dims, np.eye(3), balance=0)
    new_K *= ratio
    new_K[2, 2] = 1  # The bottom right element of the matrix should always be 1
    maps = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, dims, cv2.CV_16SC2)
    return cv2.remap(frame, *maps, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def undistort_image_video(input_path, output_path, K, D, ratio):
    """ Given a path to an image or video, undistort it
    Args:
        input_path: str
          - The input path
        output_path: str
          - The output path
        K: ndarray
          - Camera intrinsic matrix
        D: ndarray
          - Camera distortion matrix
        ratio: float
          - Ratio between training image size and undistorting image size
    """
    if input_path.split('.')[-1] in IMAGE_FILE_EXTENSIONS:
        # We have an image
        im = cv2.imread(input_path)
        undistorted = undistort_frame(im, K=K, D=D, ratio=ratio)
        cv2.imwrite(output_path, undistorted)
    else:
        # Assume it's a video and let the video undistortion method handle any checks
        undistort_individual_video(input_path, output_path, K, D, ratio)


def batch_undistort_image_video(input_dir, output_dir, K, D, ratio):
    """ Given a directory, undistorts every image and video within, sending them to output_path with the same name
    Args:
        input_dir: str
          - The input directory
        output_dir: str
          - The output directory
        K: ndarray
          - Camera intrinsic matrix
        D: ndarray
          - Camera distortion matrix
        ratio: float
          - Ratio between training image size and undistorting image size
    """
    video_files = list()
    for input_file in os.listdir(input_dir):
        if input_file.split('.')[-1] in IMAGE_FILE_EXTENSIONS:
            # We have an image
            im = cv2.imread(path.join(input_dir, input_file))
            image_name = input_file.split('.')[0]
            undistorted = undistort_frame(im, K=K, D=D, ratio=ratio)
            output_path = path.join(output_dir, input_file)
            cv2.imwrite(output_path, undistorted)
        else:
            # Assume it's a video and let the video undistortion method handle any checks
            # Collect all the videos into one place and handle them after all images have
            # been undistorted since they take longer to process
            video_in_path = path.join(input_dir, input_file)
            video_out_path = path.join(output_dir, input_file)
            video_files.append((video_in_path, video_out_path))
    # Process videos:
    for video_params in video_files:
        try:
            undistort_individual_video(*video_params, K, D, ratio)
        except Exception as e:
            print('{}: {}'.format(video_params[0], e))


def undistort_individual_video(video_path, output_path, K, D, ratio):
    """ Undistorts a single video and writes it to the provided output path
    Args:
        video_path: str
          - The video being undistorted
        output_path: str
          - Path at which the undistorted video will be saved
        K: ndarray
          - Camera intrinsic matrix
        D: ndarray
          - Camera distortion matrix
        ratio: float
          - Ratio between training image size and undistorting image size
    """
    if not output_path.endswith('.avi'):
        raise NotImplementedError('No support for video extensions other than avi at the moment')
    # Video reader w/ framerate
    vid_reader = cv2.VideoCapture(video_path)
    ret, frame = vid_reader.read()
    framerate = vid_reader.get(cv2.CAP_PROP_FPS)

    # Imdims has to be reversed because the video writer expects (width, height) rather than (rows, cols)
    im_dims = frame.shape[:2][::-1]
    is_color = len(frame.shape) == 3

    # Sometimes the framerate call fails. I believe it's just a function of how well supported
    # the video format is
    if framerate < 1:
        framerate = 30

    # Since we know the image shape ahead of time, we can precompute the maps
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, im_dims, np.eye(3), balance=0)
    new_K *= ratio
    new_K[2, 2] = 1  # The bottom right element of the matrix should always be 1
    maps = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, im_dims, cv2.CV_16SC2)

    vid_writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'DIVX'),
        30,
        im_dims,
        isColor=is_color)

    # Maintain a frame counter for the sake of displaying progress
    frame_counter = 0
    start_time = time.time()
    print('Undistorting video {}'.format(video_path))
    while ret:
        undistorted_frame = undistort_frame(frame, maps=maps)
        vid_writer.write(undistorted_frame)
        frame_counter += 1
        ret, frame = vid_reader.read()
        if (frame_counter % int(framerate * 15)) == 0:
            avg_speed = frame_counter / (time.time() - start_time)
            procd_seconds = int((frame_counter // framerate) % 60)
            procd_minutes = int(frame_counter // framerate // 60)
            if procd_minutes == 0:
                output = 'Processed {} frames ({} seconds). Avg. speed: {:.1f} frames/second.'.format(frame_counter, procd_seconds, avg_speed)
            else:
                output = 'Processed {} frames ({}:{:0>2d}). Avg. speed: {:.1f} frames/second.'.format(frame_counter, procd_minutes, procd_seconds, avg_speed)
            print(output)
    vid_writer.release()
    print('Done undistorting video {}'.format(video_path))


def run():
    """ This program takes four inputs:
      - A filepath: could be a single file (image or video) or a directory to multiple images or videos
      - A directory containing two files: K.npy and D.npy, both produced by the calibration script
      - If the images being undistorted aren't the same size as the images used for calibration, a ratio
        parameter is needed
      - An output directory or path
    """
    # Start by getting the arguments and verifying the validity of the K and D inputs
    args = get_arguments()

    try:
        K = np.load(path.join(args.parameter_path, 'K.npy'))
        D = np.load(path.join(args.parameter_path, 'D.npy'))
    except:
        print('Failed to find camera intrinsic parameters in {}. Are they properly named K.npy and D.npy?'.format(args.parameter_path))
        return

    if K.shape != (3, 3):
        print('Invalid K.npy. Try again after recomputing it.')
        return

    if D.shape != (4, 1):
        print('Possibly invalid D.npy. Try again after recomputing it.')
        return

    print('Loaded camera parameters')

    # Determine the type of input provided (file/directory) and use it to decide between batch/individual undistortion
    if not path.exists(args.input_path):
        print('Could not find the input path {}'.format(args.input_path))
        return

    if path.isdir(args.input_path):
        if not path.exists(args.output_path):
            os.mkdir(args.output_path)
        elif not path.isdir(args.output_path):
            print('Providing an input directory requires the output path to be a directory as well')
            return
        batch_undistort_image_video(args.input_path, args.output_path, K, D, args.ratio)
    elif path.isfile(args.input_path):
        if path.exists(args.output_path) and not path.isfile(args.output_path):
            input_filename = re.split('\[/\\\]', args.input_path)[-1]
            output_file_path = path.join(args.output_path, input_filename)
        else:
            output_file_path = args.output_path
        undistort_image_video(args.input_path, output_file_path, K, D, args.ratio)
    else:
        print('Invalid input path: {}'.format(args.input_path))


if __name__ == '__main__':
    run()

