import sys
sys.path.append('RAFT/core')
import re
import argparse
import os
import cv2
import glob
import numpy as np
import torch
import logging
from PIL import Image
from tqdm import tqdm
from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder
DEVICE = 'cuda'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# 配置日志
logging.basicConfig(
    filename='log.txt',  # 日志文件名
    filemode='w',        # 模式，'a' 表示追加，'w' 表示覆盖
    level=logging.INFO,  # 日志级别
    format='%(asctime)s - %(levelname)s - %(message)s'  # 日志格式
)
def load_image(frame):
    #img = np.array(imfile).astype(np.uint8)
    frame = torch.from_numpy(frame).permute(2, 0, 1).float()
    return frame[None].to(DEVICE)

def magnitude(frame1,frame2,args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        image1 = load_image(frame1)
        image2 = load_image(frame2)
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)
        flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
        # 计算光流的大小 M
        flow_up_x = flow_up[:, 0, :, :]  # 提取光流的水平分量
        flow_up_y = flow_up[:, 1, :, :]  # 提取光流的垂直分量
        flow_magnitude = torch.sqrt(flow_up_x**2 + flow_up_y**2)  # 计算光流的大小
        # 计算平均运动大小
        mean_motion = torch.mean(flow_magnitude)
    return mean_motion

def extract_frames(video_path, output_folder, interval=3, args=None):

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
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        #m = magnitude(frame1,frame2,args)
        # 每隔指定的帧数保存一次
        if frame_index % frames_per_interval == 0:
            
            #print(m)
            #output_path = os.path.join(output_folder, f"frame_{frame_index // frames_per_interval}.jpg")
            #cv2.imwrite(output_path, frame)
            frames.append(frame)
            #print(f"保存帧到 {output_path}")
            num+=1

        frame_index += 1
    for i, (frame1, frame2) in enumerate(zip(frames[:-1], frames[1:])):
        m = magnitude(frame1,frame2,args)
        if m < 100 and m > 40:
            _output_folder = f"{output_folder}_{i+1}"
            # 确保新目录存在
            os.makedirs(_output_folder, exist_ok=True)
            output_path1 = os.path.join(_output_folder, f"frame_a.jpg")
            output_path2 = os.path.join(_output_folder, f"frame_b.jpg")
            cv2.imwrite(output_path1, frame1)
            cv2.imwrite(output_path2, frame2)
            logging.info(f"Output folder: {_output_folder}  -  M: {m.item()}")
    cap.release()

# 定义参数解析器
parser = argparse.ArgumentParser()
parser.add_argument('--model', help="restore checkpoint")
#parser.add_argument('--path', help="dataset for evaluation")
parser.add_argument('--small', action='store_true', help='use small model')
parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
parser.add_argument('--alternate_corr', action='store_true', help='use efficient correlation implementation')

# 手动定义参数
args = argparse.Namespace(
    model='RAFT/models/raft-things.pth',
    #path='/mnt/sdd/zzc/humanvid_dataset/3139875-hd_1080_1920_30fps.mp4',#2785532-hd_1080_1920_25fps.mp4
    small=False,
    mixed_precision=True,
    alternate_corr=False
)


# 示例调用
videos_path = "/mnt/sdd/zzc/humanvid_dataset"  

video_files = glob.glob(os.path.join(videos_path, '**', '*.mp4'), recursive=True)
for video_path in tqdm(video_files, desc="Processing videos", unit="data"):
    output_folder = "/mnt/sdd/zzc/v2i/output_frames"  # 输出文件夹路径

    video_name = re.match(r'^(\d+)', os.path.basename(video_path)).group(1)
    output_folder = os.path.join(output_folder, video_name)

    extract_frames(video_path, output_folder, interval=2,args=args)
