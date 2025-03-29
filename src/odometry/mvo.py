import cv2
import numpy as np
import os

FOCAL = 718.856  # 초점 거리
PP = (607.1928, 185.2157)  # 주점 (cx, cy)

MAX_FRAME = 2000

SEQ = 5

PATH_SEQUENCES = f"../../sequences/{SEQ:02}/image_0/"
PATH_GROUND_TRUTH = f"../../ground_truth/{SEQ:02}.txt"
PATH_SENSOR_INFO = f"../../sensor_info/{SEQ:02}/calib.txt"

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
bf = cv2.BFMatcher(cv2.NORM_HAMMING)


def detect_keypoints(frame):
    keyPoints, descriptors = orb.detectAndCompute(frame, None)

    return keyPoints, descriptors


def get_matches(prev_descriptors, curr_descriptors):
    matches = bf.knnMatch(prev_descriptors, curr_descriptors, k=2)

    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

    return good_matches[:1500]


def adjust_pose(R, t):
    F = np.diag([1, 1, -1])
    t_adj = F @ t
    R_adj = F @ R @ F

    return R_adj, t_adj


def estimate_pose(prev_points, curr_points):
    E, _ = cv2.findEssentialMat(
        prev_points,
        curr_points,
        focal=FOCAL,
        pp=PP,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0,
    )
    _, R, t, _ = cv2.recoverPose(E, prev_points, curr_points, focal=FOCAL, pp=PP)
    R, t = adjust_pose(R, t)

    return R, t


def get_scale(gt_data, frame_number, t_gt):
    prev_x, prev_y, prev_z = t_gt.flatten()
    x, y, z = [float(gt_data[frame_number].split()[i]) for i in [3, 7, 11]]

    t_gt[:] = [[x], [y], [z]]
    scale = np.linalg.norm([x - prev_x, y - prev_y, z - prev_z])

    return scale, t_gt


def get_points_from_matches(kp1, kp2, matches):
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

    return pts1, pts2


def preprocessing():
    img1 = cv2.imread(PATH_SEQUENCES + "000000.png", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(PATH_SEQUENCES + "000001.png", cv2.IMREAD_GRAYSCALE)

    kp1, des1 = detect_keypoints(img1)
    kp2, des2 = detect_keypoints(img2)

    matches = get_matches(des1, des2)

    # draw_feature_matches(img1, kp1, img2, kp2, matches)
    pts1, pts2 = get_points_from_matches(kp1, kp2, matches)

    R_p, t_p = estimate_pose(pts1, pts2)

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


def draw_trajectory(viewer, t_gt, t_p):
    center_height = viewer.shape[0] // 2
    center_width = viewer.shape[1] // 2

    x = center_width + int(t_p[0].item())
    z = center_height - int(t_p[2].item())

    x_gt = center_width + int(t_gt[0].item())
    z_gt = center_height - int(t_gt[2].item())

    cv2.circle(viewer, (x, z), 1, (0, 0, 255), 2)
    cv2.circle(viewer, (x_gt, z_gt), 1, (255, 0, 0), 2)

    cv2.imshow("Trajectory", viewer)


def load_sensor_param():
    with open(PATH_SENSOR_INFO, "r") as f:
        line = f.readline()
        param = np.array(list((map(float, line.split()[1:])))).reshape((3, 4))

        focal = param[0, 0]
        pp = (param[0, 2], param[1, 2])

        return focal, pp


def main():
    global FOCAL
    global PP

    FOCAL, PP = load_sensor_param()

    print(FOCAL, PP)

    with open(PATH_GROUND_TRUTH, "r") as f:
        gt_data = f.readlines()

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
            PATH_SEQUENCES + f"{frame_number:06}.png",
            cv2.IMREAD_GRAYSCALE,
        )

        curr_keyPoints, curr_descriptors = detect_keypoints(frame)

        matches = get_matches(prev_descriptors, curr_descriptors)

        prev_points, curr_points = get_points_from_matches(
            prev_keyPoints, curr_keyPoints, matches
        )

        R, t = estimate_pose(prev_points, curr_points)

        scale, t_gt = get_scale(gt_data, frame_number, t_gt)

        # 이동벡터, 회전행렬 누적
        t_p = t_p + scale * R_p @ t
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
        draw_trajectory(viewer, t_gt, t_p)
        # draw_feature_matches(prev_frame, prev_keyPoints, frame, curr_keyPoints, matches)
        # draw_keypoints(frame, curr_keyPoints)
        cv2.imshow("original frame", frame)

        prev_frame = frame
        prev_keyPoints = curr_keyPoints
        prev_descriptors = curr_descriptors

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()

    draw_trajectory(viewer, t_gt, t_p)
    cv2.waitKey()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
