import cv2
import numpy as np
import os
import math


FOCAL = 718.856  # 초점 거리
PP = (607.1928, 185.2157)  # 주점 (cx, cy)

MAX_FRAME = 2000

SEQ = 0

IMAGE_FORDER_PATH = f"../../sequences/{SEQ:02}/image_0/"
POSES_PATH = f"../../poses/{SEQ:02}.txt"
PARAM_PATH = f"../../calibration/{SEQ:02}/calib.txt"

orb = cv2.ORB_create(
    nfeatures=5000,
    scaleFactor=1.2,
    nlevels=8,
    edgeThreshold=31,
    firstLevel=0,
    WTA_K=3,
    scoreType=cv2.ORB_HARRIS_SCORE,
    patchSize=31,
    fastThreshold=25,
)
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
bf = cv2.BFMatcher(cv2.NORM_HAMMING2)


def apply_orb(frame):
    keyPoints, descriptors = orb.detectAndCompute(frame, None)

    return keyPoints, descriptors


def feature_matching(prev_descriptors, curr_descriptors):
    matches = bf.knnMatch(prev_descriptors, curr_descriptors, k=2)
    # matches = bf.match(prev_descriptors, curr_descriptors)

    # matches = sorted(matches, key=lambda x: x.distance)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    return good_matches[:1500]


def essent_and_recover(curr_points, prev_points):
    E, _ = cv2.findEssentialMat(
        curr_points,
        prev_points,
        focal=FOCAL,
        pp=PP,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0,
    )
    _, R, t, _ = cv2.recoverPose(E, curr_points, prev_points, focal=FOCAL, pp=PP)

    return R, t


def get_scale(frame_number, t_gt):
    txt_pile = open(POSES_PATH)

    x_prev = float(t_gt[0])
    y_prev = float(t_gt[1])
    z_prev = float(t_gt[2])

    line = txt_pile.readlines()
    line_sp = line[frame_number].split(" ")

    x = float(line_sp[3])
    y = float(line_sp[7])
    z = float(line_sp[11])

    t_gt[0] = x
    t_gt[1] = y
    t_gt[2] = z

    txt_pile.close()

    scale = math.sqrt((x - x_prev) ** 2 + (y - y_prev) ** 2 + (z - z_prev) ** 2)

    return scale, t_gt


def preprocessing():
    img1 = cv2.imread(IMAGE_FORDER_PATH + "000000.png", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(IMAGE_FORDER_PATH + "000001.png", cv2.IMREAD_GRAYSCALE)

    kp1, des1 = apply_orb(img1)
    kp2, des2 = apply_orb(img2)

    matches = feature_matching(des1, des2)

    # draw_feature_matches(img1, kp1, img2, kp2, matches)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    R_p, t_p = essent_and_recover(pts2, pts1)

    return img2, kp2, des2, R_p, t_p


def draw_feature_matches(prev_frame, prev_keyPoints, frame, keyPoints, matches):
    feature_matching_frame = cv2.drawMatches(
        prev_frame,
        prev_keyPoints,
        frame,
        keyPoints,
        matches,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS,
    )
    cv2.imshow("Feature Matching", feature_matching_frame)


def draw_keypoints(frame, curr_keyPoints):
    frame_with_kp = cv2.drawKeypoints(frame, curr_keyPoints, None)

    cv2.imshow("Monocular Visual Odometry with Keypoints", frame_with_kp)


def draw_odom(viewer, t_gt, t_p):
    height = viewer.shape[0]
    width = viewer.shape[1]

    x_gt = int(t_gt[0]) + height // 2
    y_gt = int(t_gt[2]) + width // 2

    x = int(t_p[0]) + height // 2
    y = int(t_p[2]) + width // 2

    cv2.circle(viewer, (x, y), 1, (0, 0, 255), 2)
    cv2.circle(viewer, (x_gt, y_gt), 1, (255, 0, 0), 2)

    cv2.imshow("trajectory", viewer)


def main():
    img2, kp2, des2, R_p, t_p = preprocessing()
    prev_frame = img2
    prev_keyPoints = kp2
    prev_descriptors = des2

    t_gt = np.zeros((3, 1), dtype=np.float64)

    viewer = np.zeros((1000, 1000), dtype=np.uint8)
    viewer = cv2.cvtColor(viewer, cv2.COLOR_GRAY2BGR)

    cv2.putText(
        viewer,
        "visual odom",
        (10, 30),
        cv2.FONT_HERSHEY_COMPLEX,
        1,
        (0, 0, 255),
        1,
    )
    cv2.putText(
        viewer, "real odom", (10, 60), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1
    )

    for frame_number in range(2, MAX_FRAME):
        frame = cv2.imread(
            os.path.join(IMAGE_FORDER_PATH, f"{frame_number:06}.png"),
            cv2.IMREAD_GRAYSCALE,
        )

        curr_keyPoints, curr_descriptors = apply_orb(frame)

        matches = feature_matching(prev_descriptors, curr_descriptors)

        prev_points = np.float32(
            [prev_keyPoints[m.queryIdx].pt for m in matches]
        ).reshape(-1, 1, 2)
        curr_points = np.float32(
            [curr_keyPoints[m.trainIdx].pt for m in matches]
        ).reshape(-1, 1, 2)

        R, t = essent_and_recover(curr_points, prev_points)

        abs_scale, t_gt = get_scale(frame_number, t_gt)

        # 이동벡터, 회전행렬 누적
        t_p = t_p + abs_scale * R_p @ t
        R_p = R @ R_p

        cv2.putText(
            frame,
            f"{SEQ:02}",
            (10, 30),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            None,
            1,
        )

        # Imshow
        draw_odom(viewer, t_gt, t_p)
        # draw_feature_matches(prev_frame, prev_keyPoints, frame, curr_keyPoints, matches)
        # draw_keypoints(frame, curr_keyPoints)
        cv2.imshow("original frame", frame)

        prev_frame = frame
        prev_keyPoints = curr_keyPoints
        prev_descriptors = curr_descriptors

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()

    draw_odom(viewer, t_gt, t_p)
    cv2.waitKey()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
