import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import math
import os
import io
import pandas as pd
from tqdm import tqdm
import fastparquet as fp
import re
import json
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
import logging
# 配置日志
logging.basicConfig(
    filename='caption.txt',  # 日志文件名
    filemode='w',        # 模式，'a' 表示追加，'w' 表示覆盖
    level=logging.INFO,  # 日志级别
    format='%(asctime)s - %(levelname)s - %(message)s'  # 日志格式
)
def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        'InternVL2_5-1B': 24, 'InternVL2_5-2B': 24, 'InternVL2_5-4B': 36, 'InternVL2_5-8B': 32,
        'InternVL2_5-26B': 48, 'InternVL2_5-38B': 64, 'InternVL2_5-78B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map
# 读取图像并转换为字节数组
def image_to_bytes(image_path):
    with Image.open(image_path) as img:
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        return img_byte_arr.getvalue()
    
def read_prompt_from_file(file_path):
    """
    从TXT文件中读取提示（prompt）。
    
    :param file_path: TXT文件的路径。
    :return: 读取到的提示文本。
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            prompt = file.read()
            return prompt
    except FileNotFoundError:
        print(f"文件未找到：{file_path}")
        return None
    except Exception as e:
        print(f"读取文件时发生错误：{e}")
        return None


def get_editing_instruction(question, pixel_values, num_patches_list):
    while True:
        response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                                       num_patches_list=num_patches_list,
                                       history=None, return_history=True)
        
        # Use regex to find JSON block within code block
        match = re.search(r"```json\s*(\{.*?\})\s*```", response, re.DOTALL)
        
        if not match:
            continue  # Retry if no match found
        
        response = match.group(1)
        
        try:
            data = json.loads(response)  # Attempt to load the JSON data

            # Extract the editing instruction from the JSON response
            instruction = data.get("simple_editing_instruction") or data.get("simple_editting_instruction") or data.get("simple_edit_instruction")
            
            if instruction is None:
                continue  # Skip this iteration if instruction is None and retry
            
            return instruction  # Return the valid instruction when found
        
        except json.JSONDecodeError as e:
            # Log the error and retry
            logging.error(f"JSONDecodeError: Retrying due to error: {e}...")
            # time.sleep(2)  # Optional: Add a delay before retrying (e.g., 2 seconds)
        
        except Exception as e:
            # Catch any other exceptions (e.g., model errors, connection errors)
            logging.error(f"Error during model chat: {e}. Retrying...")
            # time.sleep(2)  # Optional: Add a delay before retrying (e.g., 2 seconds)

# If you want to load a model using multiple GPUs, please refer to the `Multiple GPUs` section.
path = '/mnt/sdf/lzh/model/InternVL2_5-8B'
device_map = split_model('InternVL2_5-8B')
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True,
    device_map=device_map).eval()#.cuda()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)



base_dir = '/mnt/cluster/disk0/lzh/output_frames_seg_blurry_clear_human'
output_file = '/mnt/cluster/disk0/lzh/dataset/edit_instructions.parquet'

# 检查输出文件是否存在，如果存在则加载已有数据
if os.path.exists(output_file):
    existing_data = pd.read_parquet(output_file)
    existing_ids = set(existing_data['ID'].values)
else:
    existing_ids = set()



# set the max number of tiles in `max_num`
# pixel_values = load_image('/home/lzh/v2i/test_images/a.jpeg', max_num=12).to(torch.float32).cuda()
generation_config = dict(max_new_tokens=1024, do_sample=True)
for dirpath, dirnames, filenames in os.walk(base_dir):
    for dirname in tqdm(dirnames, desc="Processing", unit="data"):
        
        if dirname in existing_ids:
            print(f"Skipping existing data for ID: {dirname}")
            continue  # 跳过已存在的数据
        
        dir_path = os.path.join(dirpath, dirname)
        frame_a_path = os.path.join(dir_path, 'frame_a.jpg')
        frame_b_path = os.path.join(dir_path, 'frame_b.jpg')
        frame_a_face_path = os.path.join(dir_path, 'frame_a_face.png')
        frame_b_face_path = os.path.join(dir_path, 'frame_b_face.png')
        frame_a_upper_path = os.path.join(dir_path, 'frame_a_upper_body.png')
        frame_b_upper_path = os.path.join(dir_path, 'frame_b_upper_body.png')
        frame_a_lower_path = os.path.join(dir_path, 'frame_a_lower_body.png')
        frame_b_lower_path = os.path.join(dir_path, 'frame_b_lower_body.png')
        
        pixel_values_a_face = load_image(frame_a_face_path, max_num=12).to(torch.float32).cuda()
        pixel_values_b_face = load_image(frame_b_face_path, max_num=12).to(torch.float32).cuda()
        pixel_values_faces = torch.cat((pixel_values_a_face, pixel_values_b_face), dim=0)
        num_patches_list_faces = [pixel_values_a_face.size(0), pixel_values_b_face.size(0)]
        question = read_prompt_from_file('/home/lzh/v2i/prompt/Face.txt') + '\nFace A: <image>\nFace B: <image>\n'
        response_face = get_editing_instruction(question, pixel_values_faces, num_patches_list_faces)
   
        
        pixel_values_a_upper = load_image(frame_a_upper_path, max_num=12).to(torch.float32).cuda()
        pixel_values_b_upper = load_image(frame_b_upper_path, max_num=12).to(torch.float32).cuda()
        pixel_values_upper = torch.cat((pixel_values_a_upper, pixel_values_b_upper), dim=0)
        num_patches_list_upper = [pixel_values_a_upper.size(0), pixel_values_b_upper.size(0)]
        question = read_prompt_from_file('/home/lzh/v2i/prompt/upper.txt') + '\nUpper Body A: <image>\nUpper Body B: <image>\n'
        response_upper = get_editing_instruction(question, pixel_values_upper, num_patches_list_upper)
        
        pixel_values_a_lower = load_image(frame_a_lower_path, max_num=12).to(torch.float32).cuda()
        pixel_values_b_lower = load_image(frame_b_lower_path, max_num=12).to(torch.float32).cuda()
        pixel_values_lower = torch.cat((pixel_values_a_lower, pixel_values_b_lower), dim=0)
        num_patches_list_lower = [pixel_values_a_lower.size(0), pixel_values_b_lower.size(0)]
        question = read_prompt_from_file('/home/lzh/v2i/prompt/lower.txt') + '\nLower Body A: <image>\nLower Body B: <image>\n'
        response_lower = get_editing_instruction(question, pixel_values_lower, num_patches_list_lower)
        
        
        pixel_values_a = load_image(frame_a_path, max_num=12).to(torch.float32).cuda()
        pixel_values_b = load_image(frame_b_path, max_num=12).to(torch.float32).cuda()
        pixel_values = torch.cat((pixel_values_a, pixel_values_b), dim=0)
        num_patches_list = [pixel_values_a.size(0), pixel_values_b.size(0)]
        question = read_prompt_from_file('/home/lzh/v2i/prompt/full.txt') + '\nImage A: <image>\nImage B: <image>\n' + 'Face editing instruction: ' + response_face + '\nUpper Body editing instruction: ' + response_upper + '\nLower Body editing instruction: ' + response_lower + '\n'
        instruction = get_editing_instruction(question, pixel_values, num_patches_list)

        # while True:
        #     response, history = model.chat(tokenizer, pixel_values, question, generation_config,
        #                                 num_patches_list=num_patches_list,
        #                                 history=None, return_history=True)
        #     match = re.search(r"```json\s*(\{.*?\})\s*```", response, re.DOTALL)
        #     if not match:
        #         continue
        #     response = match.group(1)
        #     try:
        #         data = json.loads(response)
        #         print(data)
        #         instruction = data.get("simple_editing_instruction") or data.get("simple_editting_instruction")or data.get("simple_edit_instruction")
        #         if instruction is None:
        #             continue  # Skip this iteration if getInstruction returned None
        #         break
        #     except json.JSONDecodeError as e:
        #     # Log the error and retry (in this case, we don't increment retries since it's infinite)
        #         logging.error(f"JSONDecodeError: Retrying due to error: {e}...")
        #         #time.sleep(2)  # Optional: Add a delay before retrying (e.g., 2 seconds)
        
        #     except Exception as e:
        #         # Catch any other exceptions (e.g., model errors, connection errors)
        #         logging.error(f"Error during model chat: {e}. Retrying...")
        #         #time.sleep(2)  # Optional: Add a delay before retrying (e.g., 2 seconds)
            
        
        
        
        
        logging.info(f"image: {dir_path}  -  instruction: {instruction}")
        # 创建数据
        data = {
            'ID':[dirname],
            'original': [image_to_bytes(frame_a_path)],
            'target': [image_to_bytes(frame_b_path)],
            'editInstruction': [instruction],
        }
        df = pd.DataFrame(data)
        if not os.path.exists(output_file):
            fp.write(output_file, df)
            continue
        fp.write(output_file, df, append=True)




# pixel_values3 = load_image('/mnt/cluster/disk0/lzh/output_frames_seg_blurry_clear_human/3044533_1/frame_a_face.png', max_num=12).to(torch.bfloat16).cuda()
# pixel_values4 = load_image('/mnt/cluster/disk0/lzh/output_frames_seg_blurry_clear_human/3044533_1/frame_b_face.png', max_num=12).to(torch.bfloat16).cuda()
# pixel_values_faces = torch.cat((pixel_values3, pixel_values4), dim=0)
# num_patches_list_faces = [pixel_values3.size(0), pixel_values4.size(0)]
# question = read_prompt_from_file('/home/lzh/v2i/prompt/Face.txt') + '\nFace-1: <image>\nFace-2: <image>\n'
# response_face, history = model.chat(tokenizer, pixel_values_faces, question, generation_config,
#                                num_patches_list=num_patches_list_faces,
#                                history=None, return_history=True)
# print(f'User: {question}\nAssistant: {response}')


# pixel_values5 = load_image('/mnt/cluster/disk0/lzh/output_frames_seg_blurry_clear_human/3044533_1/frame_a_upper_body.png', max_num=12).to(torch.bfloat16).cuda()
# pixel_values6 = load_image('/mnt/cluster/disk0/lzh/output_frames_seg_blurry_clear_human/3044533_1/frame_b_upper_body.png', max_num=12).to(torch.bfloat16).cuda()
# pixel_values_upper = torch.cat((pixel_values5, pixel_values6), dim=0)
# num_patches_list_upper = [pixel_values5.size(0), pixel_values6.size(0)]
# question = read_prompt_from_file('/home/lzh/v2i/prompt/upper.txt') + '\nUpper_Body-1: <image>\nUpper_Body-2: <image>\n'
# response_upper_body, history = model.chat(tokenizer, pixel_values_upper, question, generation_config,
#                                num_patches_list=num_patches_list_upper,
#                                history=history, return_history=True)
# print(f'User: {question}\nAssistant: {response}')



# pixel_values5 = load_image('/mnt/cluster/disk0/lzh/output_frames_seg_blurry_clear_human/3044533_1/frame_a_lower_body.png', max_num=12).to(torch.bfloat16).cuda()
# pixel_values6 = load_image('/mnt/cluster/disk0/lzh/output_frames_seg_blurry_clear_human/3044533_1/frame_a_lower_body.png', max_num=12).to(torch.bfloat16).cuda()
# pixel_values_upper = torch.cat((pixel_values5, pixel_values6), dim=0)
# num_patches_list_upper = [pixel_values5.size(0), pixel_values6.size(0)]
# question = read_prompt_from_file('/home/lzh/v2i/prompt/lower.txt') + '\nLower_Body-1: <image>\nLower_Body-2: <image>\n'
# response_lower_body, history = model.chat(tokenizer, pixel_values_upper, question, generation_config,
#                                num_patches_list=num_patches_list_upper,
#                                history=history, return_history=True)
# print(f'User: {question}\nAssistant: {response}')

# responses = "- 脸部编辑指令：" + response_face + "\n" + "- 上半身编辑指令：" + response_upper_body + "\n" + "- 下半身编辑指令：" + response_lower_body + "\n" + "- 初始编辑指令：" + response
# pixel_values1 = load_image('/mnt/cluster/disk0/lzh/output_frames_seg_blurry_clear_human/3044533_1/frame_a.jpg', max_num=12).to(torch.bfloat16).cuda()
# pixel_values2 = load_image('/mnt/cluster/disk0/lzh/output_frames_seg_blurry_clear_human/3044533_1/frame_b.jpg', max_num=12).to(torch.bfloat16).cuda()
# pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)
# num_patches_list = [pixel_values1.size(0), pixel_values2.size(0)]
# question = read_prompt_from_file('/home/lzh/v2i/prompt/Initialization.txt') + responses + '\nImage-1: <image>\nImage-2: <image>\n' + "- 改进后的编辑指令："
# response, history = model.chat(tokenizer, pixel_values, question, generation_config,
#                                num_patches_list=num_patches_list,
#                                history=history, return_history=True)
# print(f'User: {question}\nAssistant: {response}')

