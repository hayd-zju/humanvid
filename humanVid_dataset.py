import os
import requests
from urllib.parse import urlparse
from tqdm import tqdm
import time
import signal

def handler(signum, frame):
    print("\n程序被强制终止")
    exit(1)

signal.signal(signal.SIGINT, handler)

# 输入文件路径（包含视频链接的文本文件）
input_files = [
    {"file": "/mnt/cluster/disk0/lzh/datasets/pexels-horizontal-urls-new.txt", "dir": "/mnt/cluster/disk0/lzh/datasets/HumanVid/pexels-horizontal-urls-new"},
    {"file": "/mnt/cluster/disk0/lzh/datasets/pexels-vertical-urls-new.txt", "dir": "/mnt/cluster/disk0/lzh/datasets/HumanVid/pexels-vertical-urls-new"}
]

# 遍历每个输入文件
for item in input_files:
    input_file = item["file"]
    download_dir = item["dir"]

    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"文件不存在: {input_file}")
        continue

    # 创建下载目录（如果不存在则创建）
    os.makedirs(download_dir, exist_ok=True)

    # 读取文件中的每一行
    with open(input_file, "r", encoding="utf-8") as file:
        lines = file.readlines()

    # 统计有效的视频链接数量
    valid_links = [line.strip() for line in lines if line.strip().startswith("https://")]
    total_videos = len(valid_links)
    
    # 遍历每一行并下载视频
    current_video = 0  # 当前视频计数
    for line in lines:
        # 提取链接部分
        url = line.strip()
        if not url.startswith("https://"):
            # 如果不是有效的链接，跳过
            print(f"跳过无效链接: {url}")
            continue
        
        current_video += 1  # 增加当前视频计数
        # 解析链接并提取文件名
        parsed_url = urlparse(url)
        file_name = os.path.basename(parsed_url.path)

        # 下载路径
        download_path = os.path.join(download_dir, file_name)

        # 检查文件是否已经存在
        if os.path.exists(download_path):
            # 获取本地文件大小
            local_size = os.path.getsize(download_path)
            # 获取服务器文件大小
            try:
                head_response = requests.head(url, allow_redirects=True, timeout=3)
                if head_response.status_code != 200:
                    print(f"无法获取文件大小: {url}")
                    continue
                server_size = int(head_response.headers.get('content-length', 0))
                
                # 如果本地文件大小与服务器文件大小一致，跳过下载
                if local_size == server_size:
                    print(f"文件已存在且完整: {file_name}")
                    continue
                else:
                    print(f"文件存在但不完整，重新下载: {file_name}")
            except requests.exceptions.RequestException as e:
                print(f"无法获取文件大小: {url} - 错误: {str(e)}")
                continue

        # 下载视频
        try:
            # 获取文件大小
            head_response = requests.head(url, allow_redirects=True, timeout=5)
            if head_response.status_code != 200:
                print(f"无法获取文件大小: {url}")
                continue
            total_size = int(head_response.headers.get('content-length', 0))

            # 开始下载
            response = requests.get(url, stream=True, timeout=5)
            response.raise_for_status()  # 检查请求是否成功

            # 创建进度条
            progress_bar = tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=f"[{current_video}/{total_videos}] {file_name}",
                initial=0,
                ascii=True
            )

            # 保存视频
            with open(download_path, "wb") as video_file:
                downloaded_size = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        video_file.write(chunk)
                        downloaded_size += len(chunk)
                        progress_bar.update(len(chunk))  # 更新进度条
            progress_bar.close()
            print(f"成功下载: {file_name}")
        except requests.exceptions.RequestException as e:
            print(f"下载失败: {file_name} - 错误: {str(e)}")
            # 尝试删除不完整的文件
            if os.path.exists(download_path):
                os.remove(download_path)
                print(f"已删除不完整的文件: {file_name}")

        # 添加适当的延迟，避免触发服务器的频率限制
        time.sleep(1)

print("所有链接处理完成！")