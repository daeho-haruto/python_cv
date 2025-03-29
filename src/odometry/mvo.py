import cv2
import numpy as np


class VisualOdometry:
    def __init__(self, seq=0, max_frame=2000):
        self.seq = seq
        self.max_frame = max_frame
        self.path_sequences = f"../../sequences/{seq:02}/image_0/"
        self.path_ground_truth = f"../../ground_truth/{seq:02}.txt"
        self.path_sensor_info = f"../../sensor_info/{seq:02}/calib.txt"

        self.focal, self.pp = self._load_sensor_params()
        self.gt_data = self._load_ground_truth()

        self.orb = cv2.ORB_create(
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
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)

        self.viewer = self._initialize_viewer()
        self.t_gt = np.zeros((3, 1), dtype=np.float64)

        self.prev_frame, self.prev_kp, self.prev_des, self.R_p, self.t_p = (
            self._initialize_pose()
        )

    def _load_sensor_params(self):
        with open(self.path_sensor_info, "r") as f:
            param = np.array(
                list(map(float, f.readline().split()[1:])), dtype=np.float64
            ).reshape((3, 4))
        return param[0, 0], (param[0, 2], param[1, 2])

    def _load_ground_truth(self):
        with open(self.path_ground_truth, "r") as f:
            return f.readlines()

    def _initialize_viewer(self):
        viewer = np.zeros((1000, 1000, 3), dtype=np.uint8)
        cv2.putText(
            viewer, "visual odom", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1
        )
        cv2.putText(
            viewer, "real odom", (10, 60), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1
        )
        return viewer

    def _initialize_pose(self):
        img1 = cv2.imread(self.path_sequences + "000000.png", cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(self.path_sequences + "000001.png", cv2.IMREAD_GRAYSCALE)
        kp1, des1 = self._detect_features(img1)
        kp2, des2 = self._detect_features(img2)
        matches = self._get_matches(des1, des2)
        pts1, pts2 = self._get_matched_points(kp1, kp2, matches)
        R, t = self._estimate_pose(pts1, pts2)
        return img2, kp2, des2, R, t

    def _detect_features(self, frame):
        return self.orb.detectAndCompute(frame, None)

    def _get_matches(self, des1, des2, ratio=0.75, max_matches=1500):
        matches = self.bf.knnMatch(des1, des2, k=2)
        return [m for m, n in matches if m.distance < ratio * n.distance][:max_matches]

    def _get_matched_points(self, kp1, kp2, matches):
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        return pts1, pts2

    def _adjust_pose(self, R, t):
        F = np.diag([1, 1, -1])
        return F @ R @ F, F @ t

    def _estimate_pose(self, pts1, pts2):
        E, _ = cv2.findEssentialMat(
            pts1,
            pts2,
            focal=self.focal,
            pp=self.pp,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0,
        )
        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, focal=self.focal, pp=self.pp)
        return self._adjust_pose(R, t)

    def _get_scale(self, frame_number):
        prev = self.t_gt.flatten()
        values = list(map(float, self.gt_data[frame_number].split()))
        curr = np.array([values[3], values[7], values[11]])
        self.t_gt[:] = curr.reshape(3, 1)
        return np.linalg.norm(curr - prev)

    def _draw_trajectory(self):
        ch, cw = self.viewer.shape[0] // 2, self.viewer.shape[1] // 2
        x, z = cw + int(self.t_p[0]), ch - int(self.t_p[2])
        x_gt, z_gt = cw + int(self.t_gt[0]), ch - int(self.t_gt[2])
        cv2.circle(self.viewer, (x, z), 1, (0, 0, 255), 2)
        cv2.circle(self.viewer, (x_gt, z_gt), 1, (255, 0, 0), 2)
        cv2.imshow("Trajectory", self.viewer)

    def run(self):
        for i in range(2, self.max_frame):
            path = self.path_sequences + f"{i:06}.png"
            frame = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if frame is None:
                print(f"{path} not found. 종료합니다.")
                break

            kp, des = self._detect_features(frame)
            matches = self._get_matches(self.prev_des, des)
            pts1, pts2 = self._get_matched_points(self.prev_kp, kp, matches)
            R, t = self._estimate_pose(pts1, pts2)
            scale = self._get_scale(i)

            self.t_p += scale * self.R_p @ t
            self.R_p = R @ self.R_p

            cv2.putText(
                frame,
                f"{self.seq:02}",
                (10, 30),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (0, 0, 255),
                1,
            )
            self._draw_trajectory()
            cv2.imshow("Frame", frame)

            self.prev_frame, self.prev_kp, self.prev_des = frame, kp, des

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()
        self._draw_trajectory()
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    vo = VisualOdometry(seq=0, max_frame=2000)
    vo.run()
