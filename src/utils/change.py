import os
import glob

# 이미지가 저장된 폴더 경로
folder_path = "ipad_image"

new_path = "img"

# 폴더 내의 모든 jpg 파일을 찾기
image_files = glob.glob(os.path.join(folder_path, "*.jpg"))

print(image_files)
print(len(image_files))

# 파일 이름을 일괄적으로 변경
for i, old_name in enumerate(image_files, start=1):
    # 새로운 파일 이름 (예: image_1.jpg, image_2.jpg 등)
    new_name = os.path.join(new_path, f"{i}.jpg")

    # 파일 이름 변경
    os.rename(old_name, new_name)
    print(f"{old_name} -> {new_name}")
