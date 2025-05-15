import cv2
import os
def is_blurry(image, threshold=20):
    """
    判断图像是否模糊
    :param image: 输入图像
    :param threshold: 拉普拉斯方差的阈值，默认为100
    :return: True 表示图像模糊，False 表示图像清晰
    """
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 使用拉普拉斯算子计算图像的二阶导数
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # 判断是否模糊
    return laplacian_var < threshold

def extract_frames(video_path, output_folder, interval=3):
    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 加载视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频文件")
        return

    # 获取视频的帧率
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 计算每隔多少帧提取一次（根据间隔时间）
    frames_per_interval = int(fps * interval)
    num = 0
    # 初始化变量
    frame_index = 0
    while num < 3:
        ret, frame = cap.read()
        if not ret:
            break

        # 每隔指定的帧数保存一次
        if frame_index % frames_per_interval == 0:
            output_path = os.path.join(output_folder, f"frame_{frame_index // frames_per_interval}.jpg")
            while is_blurry(frame):
                ret, frame = cap.read()
                frame_index += 1
                if not ret:
                    break
            cv2.imwrite(output_path, frame)
            print(f"保存帧到 {output_path}")
            num+=1

        frame_index += 1

    cap.release()
    print("帧提取完成")




# 示例调用
video_path = "/mnt/sdd/zzc/humanvid_dataset/3139875-hd_1080_1920_30fps.mp4"  # 替换为你的视频路径
output_folder = "/home/zzc/v2i/output_frames"  # 替换为输出文件夹路径
extract_frames(video_path, output_folder, interval=1)
