import numpy as np
import cv2


def init_follow_object(frame, found):
    # setup initial location of window
    xs = [max(int(point[0][0]), 0) for point in found]
    ys = [max(int(point[0][1]), 0) for point in found]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)

    track_window = (x0, y0, x1-x0, y1-y0)

    # print track_window
    # print frame.shape[:2]
    cv2.line(frame, (x0, y0), (x1, y1), (255, 0, 0), 2)

    # set up the ROI for tracking
    roi = frame[y0:y1, x0:x1]
    hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    # Setup the termination criteria, either 10 iteration or move by atleast 1
    # pt
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    print track_window
    return frame, track_window, roi_hist, term_crit


def follow_object(frame, track_window, roi_hist, term_crit):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    # apply meanshift to get the new location
    ret, track_window = cv2.meanShift(dst, track_window, term_crit)
    print track_window

    # Draw it on image
    # pts = cv2.boxPoints(ret)
    # pts = np.int0(pts)
    # img2 = cv2.polylines(frame, [pts], True, 255, 2)
    # img2 = cv2.line(img2, (track_window[0], track_window[1]), (track_window[0] + track_window[2], track_window[1] + track_window[3]), (255, 0, 0), 2)
    x,y,w,h = track_window
    img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
    return img2, track_window


def find_object(img_to_find, background):
    MIN_MATCH_COUNT = 20
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img_to_find, None)
    kp2, des2 = sift.detectAndCompute(background, None)

    # FLANN_INDEX_KDTREE = 0
    # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    # search_params = dict(checks=50)   # or pass empty dictionary
    # search_params = dict()

    matcher = cv2.BFMatcher()
    # matcher = cv2.FlannBasedMatcher(index_params, search_params)

    # print "des1: " + str(des1)
    # print "des2: " + str(des2)
    if des1 is None:
        print "des1 is none"
        return background, None
    elif des2 is None:
        print "des2 is none"
        return background, None

    matches = matcher.knnMatch(des1, des2, k=2)
    # return background

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            good.append(m)

    dst = None
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32(
            [kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        # print M, mask
        if mask is None:
            print "mask is None"
            matchesMask = None
        else:
            matchesMask = mask.ravel().tolist()
            h, w = img_to_find.shape
            pts = np.float32(
                [[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            # print dst

            background = cv2.polylines(
                background, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    else:
        print "Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT)
        matchesMask = None

        # draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
        #                    singlePointColor = None,
        #                    matchesMask = matchesMask,
        #                    flags = 2)
        # new_img = cv2.drawMatches(
        #     img_to_find, kp1, background, kp2, good, None, **draw_params)
    return background, dst

if __name__ == "__main__":
    cv2.namedWindow("found")
    vc = cv2.VideoCapture(0)
    vc.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
    vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)
    img_cube = cv2.imread('cube.png', 0)

    if vc.isOpened():  # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    found = None
    while rval:
        frame = cv2.resize(frame, (400, 300))
        frame = cv2.flip(frame, 1)  # mirrored
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY);
        if found is None:
        # if True:
            frame, found = find_object(img_cube, frame.copy())
            if found is not None:
                (frame, track_window, roi_hist, term_crit) = init_follow_object(
                    frame, found)

        else:
            frame, track_window = follow_object(frame.copy(), track_window, roi_hist, term_crit)

        cv2.imshow("found", frame)
        # cv2.imshow("found", frame)
        rval, frame = vc.read()
        key = cv2.waitKey(30)
        if key == 27:  # exit on ESC
            break
        elif key == 32:  # space recatch
            found = None
    cv2.destroyWindow("found")
    vc.release()
