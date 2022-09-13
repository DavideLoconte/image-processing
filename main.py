
import sys
import argparse

import cv2
import numpy as np

import source
import perspective
import model

from setup import setup_video, setup_image
from detect import detect_video, detect_image


def main():
    parser = argparse.ArgumentParser(description='Social distancing detector demo')
    parser.add_argument('--image', help='Use image as source')
    parser.add_argument('--directory', help='Use a directory as source')
    parser.add_argument('--video', help='Use video as source')
    parser.add_argument('--camera', help='Use camera as source')

    # Arguments for output visualization
    parser.add_argument('--nowin', action='store_true', help='Disable output window, outputs only logs on stdout')
    parser.add_argument('--logs-frequency', help="How many frames between two consecutive logs for camera and video sources", default=10)

    # Arguments for perspective correction)
    parser.add_argument('--checkerboard-rows', help="Number of rows of the checkerboard", default=8)
    parser.add_argument('--checkerboard-cols', help="Number of columns of the checkerboard", default=11)
    parser.add_argument('--checkerboard-size', help="Length in mm of the checkerboard", default=15)
    parser.add_argument('--homography-data', help="The homography data file for setup", default='homography.bin')
    parser.add_argument('--pixel-unit', help='Use this to supply custom pixel unit and do not apply homography to sources')

    # Path of the model
    parser.add_argument('--model', help="Path of the object detector")
    parser.add_argument('setup', metavar='ACTION', type=str, nargs=1, help='SETUP, DETECT or EVALUATE')

    args = parser.parse_args(sys.argv[1:])
    sources =  sum([1 if x is not None else 0 for x in [args.image, args.directory, args.video, args.camera]])

    if sources == 0:
        print("Please select a source")
        return 1
    elif sources > 1:
        print("Please select only one source at a time")
        return 1

    if args.image:
        input_source = source.get_image(args.image)
        is_image = True
        freq = None

    if args.video:
        input_source = source.get_video(args.video)
        is_image = False
        freq = args.logs_frequency

    if args.directory:
        input_source = source.get_directory(args.directory)
        is_image = True
        freq = None

    if args.camera:
        try:
            index = int(args.camera)
        except ValueError:
            print("Please specify a integer index for camera sources")
        input_source = source.get_camera(index)
        is_image = False
        freq = args.logs_frequency

    is_video = not is_image

    if args.setup[0].lower().strip() == 'setup':
        checkerboard = int(args.checkerboard_rows), int(args.checkerboard_cols)
        size = float(args.checkerboard_size)
        destination = args.homography_data

        if is_video:
            H, d = setup_video(input_source, checkerboard, size, args.nowin)
        else:
            H, d = setup_image(input_source, checkerboard, size, args.nowin)

        if d is None:
            print("No homography data found")
        perspective.save_homography(destination, H, d)


    elif args.setup[0].lower().strip() == 'detect':
        H, d = None, None

        try:
            H, d = perspective.load_homography(args.homography_data)
        except:
            pass

        d = args.pixel_unit if args.pixel_unit is not None else d

        if d is None:
            print("Homography data not found")
            return 1

        try:
            yolo = model.get_yolo(args.model)
        except:
            yolo = None

        if yolo is None:
            print("Cannot load object detection model")
            return 1

        if is_video:
            detect_video(input_source, yolo, H, d, freq, args.nowin)
        else:
            detect_image(input_source, yolo, H, d, args.nowin)

    return 0

if __name__ == "__main__":
    sys.exit(main())
