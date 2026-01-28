import os
import cv2
import logging # 導入 logging 模組
from typing import List, Tuple

def get_video_paths(raw_path, filter_name: List[str] = None):
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
    original_videos = []
    for root, dirs, files in os.walk(raw_path):

        if filter_name is not None:
            for item in files:
                if item not in filter_name:
                    files.pop(item)
                    logging.info(f"'{item}' 找不到")

        for file in files:
            if any(file.lower().endswith(ext) for ext in video_extensions):
                full_path = os.path.join(root, file)
                try:
                    cap = cv2.VideoCapture(full_path)
                    if cap.isOpened():
                        original_videos.append(full_path)
                    cap.release()
                except Exception as e:
                    logging.error(f"無法讀取文件 {full_path}: {str(e)}")
    return original_videos