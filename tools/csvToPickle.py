import os
import glob
import numpy as np
import pandas as pd
import pickle
import ast
import logging # 導入 logging 模組
from sklearn.model_selection import train_test_split
import math # Needed for ceil

def segment_and_save_csvs(input_csv_path, output_segment_dir, segment_length, segment_interval=None,
                         min_segment_length=None, remove_partial_segments=False):
    """
    Segments a single keypoint CSV file into smaller CSV files based on frame ranges.

    Args:
        input_csv_path (str): Path to the input keypoint CSV file.
        output_segment_dir (str): Directory to save the segmented CSV files.
        segment_length (int): The desired length of each segment in frames.
        segment_interval (int, optional): The interval between the start of segments.
                                          If None, it defaults to segment_length (no overlap).
        min_segment_length (int, optional): The minimum length of segments to keep.
                                           If None, defaults to 150 frames.
        remove_partial_segments (bool, optional): Whether to remove segments that are above
                                                 min_segment_length but below segment_length.
                                                 If True, only keeps segments with exactly segment_length frames.
                                                 If False, keeps segments >= min_segment_length.

    Returns:
        list: A list of tuples, where each tuple contains the path to a saved segment CSV
              and the original input CSV filename.
    """
    # 設置預設的最小片段長度
    if min_segment_length is None:
        min_segment_length = 150
    
    if segment_length is None:
        # 如果沒有提供 segment_length，檢查原始文件是否符合最小長度要求
        try:
            df = pd.read_csv(input_csv_path)
            if df.empty:
                logging.warning(f"跳過空文件的處理: {input_csv_path}")
                return []
            
            total_frames = len(df)
            if total_frames < min_segment_length:
                logging.info(f"跳過過短文件 {input_csv_path}: 只有 {total_frames} 幀 (少於最小要求 {min_segment_length})")
                return []
            
            original_filename = os.path.basename(input_csv_path)
            return [(input_csv_path, original_filename)]
        except Exception as e:
            logging.error(f"讀取文件時出錯 {input_csv_path}: {str(e)}")
            return []

    try:
        df = pd.read_csv(input_csv_path)
        if df.empty:
            logging.warning(f"Skipping segmentation for empty file: {input_csv_path}")
            return []

        # Determine the frame range from the data
        # Assuming 'frame_id' column exists and is sorted
        all_frames = df['frame_id'].tolist()
        if not all_frames:
             logging.warning(f"Skipping segmentation for file with no frame_id: {input_csv_path}")
             return []

        track_start = all_frames[0]
        track_end = all_frames[-1]
        total_frames_in_csv = len(all_frames)

        # 如果總幀數少於最小片段長度，直接跳過
        if total_frames_in_csv < min_segment_length:
            logging.info(f"跳過過短文件 {input_csv_path}: 只有 {total_frames_in_csv} 幀 (少於最小要求 {min_segment_length})")
            return []
        
        # 如果 segment_length 大於總幀數，且符合最小長度要求，則不進行切割
        if segment_length > total_frames_in_csv:
            if remove_partial_segments:
                # 如果啟用了移除部分片段，且總幀數小於完整片段長度，則跳過
                logging.info(f"跳過部分片段 {input_csv_path}: 總幀數 {total_frames_in_csv} 小於完整片段長度 {segment_length}")
                return []
            else:
                # 否則保留這個符合最小長度要求的文件
                logging.info(f"保留非完整片段 {input_csv_path}: segment_length ({segment_length}) > total_frames ({total_frames_in_csv})")
                original_filename = os.path.basename(input_csv_path)
                return [(input_csv_path, original_filename)]


        # If interval is None, set it to segment_length
        if segment_interval is None:
            segment_interval = segment_length

        segments_info = []
        original_filename = os.path.basename(input_csv_path)
        base_name_without_ext = os.path.splitext(original_filename)[0]

        # Generate segments
        current_start_frame_idx = 0 # Index in the dataframe
        segment_count = 0

        while current_start_frame_idx < total_frames_in_csv:
            # Calculate the actual frame ID for the start of the segment
            current_start_frame_id = all_frames[current_start_frame_idx]
            # Calculate the actual frame ID for the end of the segment
            # We need segment_length frames, so the end index is start_idx + segment_length - 1
            current_end_frame_idx = current_start_frame_idx + segment_length - 1

            # 檢查片段是否超出可用幀的範圍
            if current_end_frame_idx >= total_frames_in_csv:
                # 處理最後一個不完整的片段
                remaining_frames = total_frames_in_csv - current_start_frame_idx
                
                if remaining_frames >= min_segment_length:
                    if remove_partial_segments:
                        # 如果啟用了移除部分片段，且剩餘幀數小於完整片段長度，則跳過
                        logging.info(f"跳過最後的部分片段: 剩餘 {remaining_frames} 幀 (小於完整片段長度 {segment_length})")
                        break
                    else:
                        # 否則保留這個符合最小長度要求的片段
                        current_end_frame_idx = total_frames_in_csv - 1
                        current_end_frame_id = all_frames[current_end_frame_idx]
                        
                        # 提取片段數據
                        segment_df = df.iloc[current_start_frame_idx : current_end_frame_idx + 1].copy()
                        
                        # 創建片段文件名
                        segment_filename = f"{base_name_without_ext}_seg{segment_count}_{current_start_frame_id}-{current_end_frame_id}.csv"
                        output_path = os.path.join(output_segment_dir, segment_filename)
                        
                        # 保存片段CSV
                        segment_df.to_csv(output_path, index=False)
                        segments_info.append((output_path, original_filename))
                        segment_count += 1
                        
                        logging.info(f"保留最後的部分片段: {remaining_frames} 幀")
                else:
                    logging.info(f"跳過最後的過短片段: 剩餘 {remaining_frames} 幀 (少於最小要求 {min_segment_length})")
                
                break # 處理完最後一個片段後結束

            current_end_frame_id = all_frames[current_end_frame_idx]

            # Extract the segment data
            segment_df = df.iloc[current_start_frame_idx : current_end_frame_idx + 1].copy()

            # Create segment filename
            segment_filename = f"{base_name_without_ext}_seg{segment_count}_{current_start_frame_id}-{current_end_frame_id}.csv"
            output_path = os.path.join(output_segment_dir, segment_filename)

            # Save the segment CSV
            segment_df.to_csv(output_path, index=False)
            segments_info.append((output_path, original_filename))

            # Move to the next segment start index based on interval
            # Find the index of the frame that is segment_interval frames after the current_start_frame_id
            next_start_frame_id = current_start_frame_id + segment_interval
            try:
                # Find the index of the next_start_frame_id in the all_frames list
                current_start_frame_idx = all_frames.index(next_start_frame_id)
            except ValueError:
                # If the exact frame ID is not found (due to skipped frames), find the closest one or stop
                # A simple approach is to just move the index by the interval if frames are contiguous,
                # but since frames might be skipped, finding the index is safer.
                # If the next start frame ID is not in the list, it means we've gone past the available frames.
                break # Stop if the next segment start frame ID is not found

            segment_count += 1

        logging.info(f"Segmented {original_filename} into {segment_count} segments.")
        return segments_info

    except Exception as e:
        logging.error(f"Error segmenting {input_csv_path}: {str(e)}")
        return []


def load_keypoint_csv(file_path, target_frames=None, img_shape=(1080, 1920), selected_parts=None,
                     min_segment_length=None, keep_original_dimensions=True):
    """Load keypoint data from a CSV file.

    Args:
        file_path: Path to CSV file
        target_frames: Target number of frames (deprecated, use min_segment_length instead)
        img_shape: Image shape (width, height)
        selected_parts: List of body parts to include. If None, includes all parts.
            Valid parts: ['Nose', 'LEye', 'REye', 'LEar', 'REar',
                        'LShoulder', 'RShoulder', 'LElbow', 'RElbow',
                        'LWrist', 'RWrist', 'LHip', 'RHip',
                        'LKnee', 'RKnee', 'LAnkle', 'RAnkle']
        min_segment_length: Minimum number of frames required for a CSV to be processed
        keep_original_dimensions: If True, keep original 17 keypoints with zero padding for unselected parts.
                                If False, only keep selected parts (reduces dimensions).
    """
    try:
        df = pd.read_csv(file_path)

        # 檢查文件是否有足夠的幀數
        effective_min_length = min_segment_length if min_segment_length is not None else (target_frames if target_frames is not None else 150)
        if len(df) < effective_min_length:
            logging.info(f"跳過 {file_path}: 只有 {len(df)} 幀 (少於最小要求 {effective_min_length})")
            return None

        # Define all possible parts in standard order
        all_parts = [
            'Nose', 'LEye', 'REye', 'LEar', 'REar',
            'LShoulder', 'RShoulder', 'LElbow', 'RElbow',
            'LWrist', 'RWrist', 'LHip', 'RHip',
            'LKnee', 'RKnee', 'LAnkle', 'RAnkle'
        ]

        # If no selected parts specified, use all parts
        if selected_parts is None:
            selected_parts = all_parts

        # Create mask for selected parts
        part_mask = [1 if part in selected_parts else 0 for part in all_parts]

        # Initialize arrays
        num_frames = len(df)
        keypoint_array = np.zeros((1, num_frames, 17, 2))  # [M=1, T=frames, V=17, C=2]
        keypoint_score = np.zeros((1, num_frames, 17))     # Start with 0 confidence

        # Set confidence to 1 for selected parts
        for kp_idx, mask in enumerate(part_mask):
            if mask:
                keypoint_score[0, :, kp_idx] = 1.0

        # Process each frame
        for frame_idx, row in df.iterrows():
            if frame_idx >= num_frames:
                break

            # Parse the keypoints string into a list of coordinates
            try:
                keypoints_str = row['keypoints']
                keypoints_list = ast.literal_eval(keypoints_str)

                # Fill the keypoint array only for selected parts
                for kp_idx, kp in enumerate(keypoints_list):
                    if part_mask[kp_idx]:  # Ensure within bounds and part is selected
                        keypoint_array[0, frame_idx, kp_idx, 0] = float(kp[0])  # x coordinate
                        keypoint_array[0, frame_idx, kp_idx, 1] = float(kp[1])  # y coordinate
            except Exception:
                # Use individual columns as fallback
                for kp_idx, kp_name in enumerate([
                    'Nose', 'LEye', 'REye', 'LEar', 'REar',
                    'LShoulder', 'RShoulder', 'LElbow', 'RElbow',
                    'LWrist', 'RWrist', 'LHip', 'RHip',
                    'LKnee', 'RKnee', 'LAnkle', 'RAnkle'
                ]):
                    x_col = f"{kp_name}_x"
                    y_col = f"{kp_name}_y"

                    if x_col in row and y_col in row and part_mask[kp_idx]:
                        x_val = float(row[x_col])
                        y_val = float(row[y_col])
                        keypoint_array[0, frame_idx, kp_idx, 0] = x_val
                        keypoint_array[0, frame_idx, kp_idx, 1] = y_val

        return {
            'keypoint': keypoint_array,
            'keypoint_score': keypoint_score,
            'total_frames': num_frames,
            'img_shape': (img_shape[1], img_shape[0]),  # (height, width) in mmaction format
            'original_shape': (img_shape[1], img_shape[0])  # (height, width) in mmaction format
        }

    except Exception as e:
        logging.error(f"Error loading {file_path}: {str(e)}")
        return None

def main(keypoints_folder, labels_file_path, label_name, output_dir, img_shape=(1080, 1920),
         selected_parts=None, segment_length=None, segment_interval=None,
         min_segment_length=None, remove_partial_segments=False):
    """
    Processes keypoint CSV files, segments them if specified, and creates a COCO format pickle dataset.

    Args:
        keypoints_folder (str): Folder containing the input keypoint CSV files.
        labels_file_path (str): Path to the CSV file with video labels.
        label_name (str): The name of the column in the labels file to use as the label.
        output_dir (str): Directory to save the output pickle file and segmented CSVs.
        img_shape (tuple, optional): The shape of the images (width, height). Defaults to (1080, 1920).
        selected_parts (list, optional): List of body parts to include. If None, includes all parts.
        segment_length (int, optional): The desired length of each segment in frames.
                                        If None, no segmentation is performed.
        segment_interval (int, optional): The interval between the start of segments.
                                          If None and segment_length is provided, it defaults to segment_length.
        min_segment_length (int, optional): The minimum length of segments to keep.
                                           If None, defaults to 150 frames.
        remove_partial_segments (bool, optional): Whether to remove segments that are above
                                                 min_segment_length but below segment_length.
                                                 If True, only keeps segments with exactly segment_length frames.
                                                 If False, keeps segments >= min_segment_length.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define output file paths
    output_path = os.path.join(output_dir, 'train.pkl')
    segmented_keypoints_folder = os.path.join(output_dir, 'segments') # New directory for segmented CSVs
    os.makedirs(segmented_keypoints_folder, exist_ok=True) # Create the new directory

    # Load labels data
    labels_df = pd.read_csv(labels_file_path)
    labels_dict = dict(zip(labels_df['keypoint_file'], labels_df[label_name]))
    id_dict = dict(zip(labels_df['keypoint_file'], labels_df['id']))

    # Find all original keypoint CSV files
    original_keypoint_files = glob.glob(os.path.join(keypoints_folder, "*.csv"))
    logging.info(f"Found {len(original_keypoint_files)} original keypoint CSV files")

    # Segment files and collect info for all resulting CSVs (original or segmented)
    all_processed_files_info = [] # List of (file_path, original_filename) tuples

    for original_file_path in original_keypoint_files:
        original_filename = os.path.basename(original_file_path)

        # Skip if no label is available for the original file
        if original_filename not in labels_dict:
            logging.info(f"Skipping {original_filename}: No label found")
            continue

        # 分割原始文件並獲取所有結果片段文件的信息
        segments_info = segment_and_save_csvs(
            original_file_path,
            segmented_keypoints_folder,
            segment_length,
            segment_interval,
            min_segment_length,
            remove_partial_segments
        )
        all_processed_files_info.extend(segments_info)

    logging.info(f"Total processed CSV files (including segments): {len(all_processed_files_info)}")

    # Process files and create annotations
    annotations = []
    file_source_map = {}  # Map to track which source each file belongs to

    for processed_file_path, original_filename in all_processed_files_info:
        processed_file_name = os.path.basename(processed_file_path)

        # 從處理後的文件載入關鍵點數據（片段或原始文件）
        logging.info(f"從 {processed_file_name} 載入數據...")
        keypoint_data = load_keypoint_csv(processed_file_path, segment_length, img_shape, selected_parts, min_segment_length)
        if keypoint_data is None:
            continue

        # Create annotation entry
        # Use the processed file name (without extension) as the frame_dir/file_id
        file_id = processed_file_name.replace('.csv', '')
        # Use the original filename to look up the source_id and label
        source_id = id_dict.get(original_filename)
        label = labels_dict.get(original_filename)

        if source_id is None or label is None:
             logging.warning(f"Could not find source_id or label for original file {original_filename} associated with segment {processed_file_name}. Skipping.")
             continue


        annotation = {
            'frame_dir': file_id,
            'source_id': source_id,
            'label': int(label),
            'img_shape': keypoint_data['img_shape'],
            'original_shape': keypoint_data['original_shape'],
            'total_frames': keypoint_data['total_frames'],
            'keypoint': keypoint_data['keypoint'],
            'keypoint_score': keypoint_data['keypoint_score']
        }

        annotations.append(annotation)
        file_source_map[file_id] = source_id

    # Group videos by source
    sources = {}
    for file_id, source_id in file_source_map.items():
        if source_id not in sources:
            sources[source_id] = []
        sources[source_id].append(file_id)

    # Split sources into train and val sets (80:20)
    source_ids = list(sources.keys())
    # print(source_ids)
    train_sources, val_sources = train_test_split(source_ids, test_size=0.2, random_state=42)

    train_videos = [vid for source in train_sources for vid in sources[source]]
    val_videos = [vid for source in val_sources for vid in sources[source]]

    # Create combined dictionary
    combined_dict = {
        'split': {
            'train': train_videos,
            'val': val_videos,
            'video_source_train': train_sources,
            'video_source_val': val_sources,
        },
        'annotations': annotations  # Include all annotations
    }

    # print(combined_dict)

    # Save as pickle file
    with open(output_path, 'wb') as f:
        pickle.dump(combined_dict, f)

    # 統計 train 和 val 資料集中 clips 和 sources 的 label_name 每個類別的數量
    def get_label_stats(video_list, source_list, annotations, labels_df, label_name):
        """計算指定視頻列表和來源列表的標籤統計"""
        # 建立 source_id 到 label 的映射
        source_label_map = dict(zip(labels_df['id'], labels_df[label_name]))
        
        # 統計 clips 的標籤分布
        clip_labels = []
        for annotation in annotations:
            if annotation['frame_dir'] in video_list:
                clip_labels.append(annotation['label'])
        
        # 統計 sources 的標籤分布
        source_labels = []
        for source_id in source_list:
            if source_id in source_label_map:
                source_labels.append(int(source_label_map[source_id]))
        
        # 計算每個類別的數量
        from collections import Counter
        clip_counter = Counter(clip_labels)
        source_counter = Counter(source_labels)
        
        return clip_counter, source_counter
    
    # 獲取統計數據
    train_clip_stats, train_source_stats = get_label_stats(train_videos, train_sources, annotations, labels_df, label_name)
    val_clip_stats, val_source_stats = get_label_stats(val_videos, val_sources, annotations, labels_df, label_name)
    
    # 獲取所有可能的標籤值
    all_labels = set()
    for annotation in annotations:
        all_labels.add(annotation['label'])
    all_labels = sorted(list(all_labels))
    
    logging.info(f"Created COCO format dataset:")
    logging.info(f"  - Combined: {len(annotations)} clips ({len(train_videos)} train, {len(val_videos)} val)")
    logging.info(f"  - Sources: {len(source_ids)} total ({len(train_sources)} in train, {len(val_sources)} in val)")
    
    # 輸出詳細的標籤統計
    logging.info(f"\n=== {label_name} 標籤統計 ===")
    
    # Train 資料集統計
    logging.info(f"\nTrain 資料集:")
    logging.info(f"  Clips 標籤分布:")
    for label in all_labels:
        count = train_clip_stats.get(label, 0)
        logging.info(f"    標籤 {label}: {count} clips")
    logging.info(f"  總計: {sum(train_clip_stats.values())} clips")
    
    logging.info(f"  Sources 標籤分布:")
    for label in all_labels:
        count = train_source_stats.get(label, 0)
        logging.info(f"    標籤 {label}: {count} sources")
    logging.info(f"  總計: {sum(train_source_stats.values())} sources")
    
    # Val 資料集統計
    logging.info(f"\nVal 資料集:")
    logging.info(f"  Clips 標籤分布:")
    for label in all_labels:
        count = val_clip_stats.get(label, 0)
        logging.info(f"    標籤 {label}: {count} clips")
    logging.info(f"  總計: {sum(val_clip_stats.values())} clips")
    
    logging.info(f"  Sources 標籤分布:")
    for label in all_labels:
        count = val_source_stats.get(label, 0)
        logging.info(f"    標籤 {label}: {count} sources")
    logging.info(f"  總計: {sum(val_source_stats.values())} sources")
    
    # 總體統計
    logging.info(f"\n總體統計:")
    total_clip_stats = {}
    total_source_stats = {}
    for label in all_labels:
        total_clip_stats[label] = train_clip_stats.get(label, 0) + val_clip_stats.get(label, 0)
        total_source_stats[label] = train_source_stats.get(label, 0) + val_source_stats.get(label, 0)
    
    logging.info(f"  所有 Clips 標籤分布:")
    for label in all_labels:
        count = total_clip_stats[label]
        logging.info(f"    標籤 {label}: {count} clips")
    logging.info(f"  總計: {sum(total_clip_stats.values())} clips")
    
    logging.info(f"  所有 Sources 標籤分布:")
    for label in all_labels:
        count = total_source_stats[label]
        logging.info(f"    標籤 {label}: {count} sources")
    logging.info(f"  總計: {sum(total_source_stats.values())} sources")
    
    logging.info(f"\nSaved to {output_path}")

if __name__ == "__main__":
    # import argparse

    # parser = argparse.ArgumentParser(description='Convert keypoint CSVs to COCO format')
    # parser.add_argument('--keypoints_folder', required=True, help='Folder with keypoint CSVs')
    # parser.add_argument('--labels_file', required=True, help='CSV file with video labels')
    # parser.add_argument('--output', default='action_dataset.pkl', help='Output pickle file')
    # parser.add_argument('--segment_length', type=int, default=None, help='Length of each segment in frames')
    # parser.add_argument('--segment_interval', type=int, default=None, help='Interval between segment starts in frames')
    # parser.add_argument('--selected_parts', nargs='+', default=None, help='List of body parts to include')

    # args = parser.parse_args()
    import sys
    import os
    sys.path.append(os.path.abspath('../src'))

    from config import ConfigManager
    from utils.logger_config import setup_logging # 導入 setup_logging 函數

    setup_logging() # 初始化 logger

    config_manager = ConfigManager("config.yaml")
    config = config_manager.config.get("data")

    keypoints_folder = "../outputs/csv"
    metadata_file = "../outputs/2025_normal_keypoint_analysis_metadata.csv"
    label_name = "age_months_range"
    # label_name = "result_months_range"
    output = "../outputs/pickle"
    selected_parts = ['Nose', 'LHip', 'RHip', 'LKnee', 'RKnee', 'LAnkle', 'RAnkle']

    # 從配置文件讀取片段參數
    # segment_length = config.get("segment_length")
    # segment_interval = config.get("segment_interval")
    segment_length = 300
    segment_interval = None
    min_segment_length = 150  # 設定最小片段長度（原本的過短片段剔除功能）
    remove_partial_segments = False  # 設定是否移除部分片段（長度高於下限但不滿segment_length的片段）
    
    selected_parts_config = config.get("selected_parts", None) # 從配置文件讀取selected_parts

    # 如果配置文件中有selected_parts，則使用配置文件中的，否則使用預設列表
    if selected_parts_config is not None:
        selected_parts = selected_parts_config

    # 從配置文件讀取新增的參數
    # if "min_segment_length" in config:
    #     min_segment_length = config.get("min_segment_length")
    if "remove_partial_segments" in config:
        remove_partial_segments = config.get("remove_partial_segments")

    main(keypoints_folder, metadata_file, label_name, output,
         img_shape=(1080, 1920), # 保持預設img_shape或根據需要從配置文件讀取
        #  selected_parts=selected_parts,
         segment_length=segment_length,
         segment_interval=segment_interval,
         min_segment_length=min_segment_length,
         remove_partial_segments=remove_partial_segments)
