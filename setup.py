import time

import perspective
import visualize
import cv2

def is_oriented(frame, corners):

    if frame is None or corners is None:
        return False

    if corners[0][1] > corners[1][1]:
        return False
    if abs(corners[1][1] - corners[3][1]) > frame.shape[0] * 0.25:
        print("Out of bound")
        return False

    return True

def setup_video(source, checkerboard, size, nowin):
    print("Setup through video source")
    print("Please lay down the checkerboard pattern and align it with the camera")
    print("When the checkerboard is found it will be displayed on the screen")
    print("Rotate the checkerboard so that the rows are parallel to the top and bottom image borders")
    start = time.time()
    end = time.time()
    delta = end - start

    # Logging
    found = False
    lost = False

    for frame in source:
        points = perspective.find_checkerboard(frame, checkerboard)

        if points is None:
            if not nowin:
                cv2.imshow('setup', frame)
                cv2.waitKey(1)
            start = time.time()
            continue

        corners = perspective.get_corners(points, checkerboard)

        if is_oriented(frame, corners):

            if not found:
                print("Found checkerboard pattern")
                found = True
                lost = False

            frame = cv2.drawChessboardCorners(frame, checkerboard, points, True)
            end = time.time()
            delta = end - start
            if delta > 2:
                cv2.destroyAllWindows()
                return perspective.single_checkerboard_calibration(points, checkerboard, size)
        else:

            if not lost:
                lost = True
                found = False
                print("No checkerboard pattern found")

            start = time.time()
        if not nowin:
            cv2.imshow('setup', frame)
            cv2.waitKey(1)
    return None, None

def setup_image(source, checkerboard, size, nowin):
    print("Setup through image sources")
    i = 0
    for frame in source:
        i += 1
        points = perspective.find_checkerboard(frame, checkerboard)
        if points is None:
            continue
        corners = perspective.get_corners(points, checkerboard)
        bottom_right = corners[-1][1]
        bottom_left = corners[-1][1]

        if abs(bottom_right - bottom_left) < frame.shape[1] * 0.05:
            print(f"Found checkerboard in {i} image")
            if not is_oriented(frame, corners):
                print("Warning!!! Checkerboard is not correctly oriented")
            frame = cv2.drawChessboardCorners(frame, (checkerboard[0] - 1, checkerboard[1] - 1), points, True)
            if not nowin:
                cv2.imshow('setup', frame)
                cv2.waitKey(0)
            return perspective.single_checkerboard_calibration(points, checkerboard, size)
    print("No checkerboard found")
    return None, None