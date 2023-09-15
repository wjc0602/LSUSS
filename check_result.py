import os
import filecmp

dir1 = '/data/comptition/jittor3/Jittor2_test_a/test'
dir2 = '/data/comptition/jittor3/PASS-jittor/results/001_baseline_resnet18_e400/test'

# 获取两个文件夹中所有图片的路径
def get_image_paths(dir_path):
    image_paths = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith('.JPEG') or file.endswith('.png'):
                image_paths.append(file.split('.')[0])
    return image_paths

# 获取A和B文件夹中所有图片的路径
image_paths_a = get_image_paths(dir1)
image_paths_b = get_image_paths(dir2)
image_paths_a.sort()
image_paths_b.sort()
# 找出A中存在但是不中不存在的图片路径
diff_a = set(image_paths_a) - set(image_paths_b)

# 找出B中存在但是A中不存在的图片路径
diff_b = set(image_paths_b) - set(image_paths_a)

print('A中存在但是不中不存在的图片路径：')
for path in diff_a:
    print(path)

print('B中存在但是A中不存在的图片路径：')
for path in diff_b:
    print(path)
