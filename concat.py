from PIL import Image
import os
from tqdm import tqdm

def create_side_by_side_image(image_path1, image_path2, output_path):
    # 打开两张图片
    image1 = Image.open(image_path1)
    image2 = Image.open(image_path2)

    # 获取两张图片的尺寸
    width1, height1 = image1.size
    width2, height2 = image2.size

    # 确定拼接后图片的尺寸
    total_width = width1 + width2
    max_height = max(height1, height2)

    # 创建一个新的空白图片用于拼接
    new_image = Image.new('RGB', (total_width, max_height))

    # 将第一张图片粘贴到新图片的左侧
    new_image.paste(image1, (0, 0))

    # 将第二张图片粘贴到新图片的右侧
    # 需要根据第一张图片的高度来调整第二张图片的粘贴位置
    new_image.paste(image2, (width1, max(0, height1 - height2)))

    # 保存拼接后的图片
    new_image.save(output_path)

def process_directory(base_dir):
    # 遍历基础目录下的所有子目录
    for dirpath, dirnames, filenames in os.walk(base_dir):
        for dirname in tqdm(dirnames,desc="Processing videos", unit="data"):
        # 检查目录中是否有frame_a.jpg和frame_b.jpg
            dir_path = os.path.join(dirpath, dirname)
            if 'frame_a.jpg' in os.listdir(dir_path) and 'frame_b.jpg' in os.listdir(dir_path):
                frame_a_path = os.path.join(dir_path, 'frame_a.jpg')
                frame_b_path = os.path.join(dir_path, 'frame_b.jpg')
                output_path = os.path.join(dir_path, 'diptych.jpg')
                create_side_by_side_image(frame_a_path, frame_b_path, output_path)
            #print(f"Processed {dirname}")

# 使用示例
base_dir = '/mnt/sdd/zzc/v2i/output_frames'
process_directory(base_dir)