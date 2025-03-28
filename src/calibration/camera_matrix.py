import numpy as np
import cv2
import glob

# 디바이스별 설정 (ipad, paper 등)
device_config = {
    "ipad": {"chess": (10, 7), "path": "image/ipad/*.jpg"},
    "standard": {"chess": (9, 6), "path": "image/standard/*.jpg"},
    "standard2": {"chess": (9, 6), "path": "image/standard2/*.jpg"},
}

# 사용할 장치 설정 (현재 'ipad' 선택)
board = device_config["standard2"]

# 체스보드 인식 조건 설정
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 3D 객체 점 (체스보드의 실제 물리적 점들)
objp = np.zeros((board["chess"][0] * board["chess"][1], 3), np.float32)
objp[:, :2] = np.mgrid[0 : board["chess"][0], 0 : board["chess"][1]].T.reshape(-1, 2)

# 2D 이미지 점과 3D 객체 점을 저장할 리스트
objpoints = []
imgpoints = []

# 이미지 파일 목록 불러오기
images = glob.glob(board["path"])

# 체스보드 코너 찾기
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, board["chess"], None)

    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # 체스보드 코너를 이미지에 그리기
        cv2.drawChessboardCorners(img, board["chess"], corners2, ret)
        cv2.imshow("img", img)
        cv2.waitKey(500)
    else:
        print(f"체스보드 찾기 실패: {fname}")

cv2.destroyAllWindows()

# 카메라 캘리브레이션 수행
if objpoints and imgpoints:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    print("Camera Matrix:\n", mtx)
    print("Distortion Coefficients:\n", dist)

    # 캘리브레이션 결과 저장 (선택사항)
    np.savez("calibration_data.npz", mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
else:
    print("체스보드 코너를 찾지 못했습니다. 패턴 크기 또는 이미지 품질을 확인하세요.")
