import numpy as np
import cv2

# 저장된 캘리브레이션 데이터 불러오기
with np.load("calibration_data.npz") as data:
    print(data)
    camera_matrix = data["mtx"]
    dist_coeffs = data["dist"]

print("Camera Matrix:\n", camera_matrix)
print("\nDistortion Coefficients:\n", dist_coeffs)

# 이미지 로드
img = cv2.imread("image/standard2/1.jpg")

# 왜곡 제거
h, w = img.shape[:2]
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
    camera_matrix, dist_coeffs, (w, h), 1, (w, h)
)

# 왜곡 제거 수행
undistorted_img = cv2.undistort(
    img, camera_matrix, dist_coeffs, None, new_camera_matrix
)

# 결과 출력
cv2.imshow("Original Image", img)
cv2.imshow("Undistorted Image", undistorted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
