import os
import glob
import numpy as np
import pandas as pd
import pickle
import ast
import logging
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def segment_and_save_csvs(input_csv_path, output_segment_dir, segment_length, segment_interval=None,
                         min_segment_length=None, remove_partial_segments=False):
    if min_segment_length is None:
        min_segment_length = 150
    
    if segment_length is None:
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

        all_frames = df['frame_id'].tolist()
        if not all_frames:
             logging.warning(f"Skipping segmentation for file with no frame_id: {input_csv_path}")
             return []

        total_frames_in_csv = len(all_frames)

        if total_frames_in_csv < min_segment_length:
            logging.info(f"跳過過短文件 {input_csv_path}: 只有 {total_frames_in_csv} 幀 (少於最小要求 {min_segment_length})")
            return []
        
        if segment_length > total_frames_in_csv:
            if remove_partial_segments:
                logging.info(f"跳過部分片段 {input_csv_path}: 總幀數 {total_frames_in_csv} 小於完整片段長度 {segment_length}")
                return []
            else:
                logging.info(f"保留非完整片段 {input_csv_path}: segment_length ({segment_length}) > total_frames ({total_frames_in_csv})")
                original_filename = os.path.basename(input_csv_path)
                return [(input_csv_path, original_filename)]

        if segment_interval is None:
            segment_interval = segment_length

        segments_info = []
        original_filename = os.path.basename(input_csv_path)
        base_name_without_ext = os.path.splitext(original_filename)[0]

        current_start_frame_idx = 0
        segment_count = 0

        while current_start_frame_idx < total_frames_in_csv:
            current_start_frame_id = all_frames[current_start_frame_idx]
            current_end_frame_idx = current_start_frame_idx + segment_length - 1

            if current_end_frame_idx >= total_frames_in_csv:
                remaining_frames = total_frames_in_csv - current_start_frame_idx
                
                if remaining_frames >= min_segment_length:
                    if remove_partial_segments:
                        logging.info(f"跳過最後的部分片段: 剩餘 {remaining_frames} 幀 (小於完整片段長度 {segment_length})")
                        break
                    else:
                        current_end_frame_idx = total_frames_in_csv - 1
                        current_end_frame_id = all_frames[current_end_frame_idx]
                        
                        segment_df = df.iloc[current_start_frame_idx : current_end_frame_idx + 1].copy()
                        
                        segment_filename = f"{base_name_without_ext}_seg{segment_count}_{current_start_frame_id}-{current_end_frame_id}.csv"
                        output_path = os.path.join(output_segment_dir, segment_filename)
                        
                        segment_df.to_csv(output_path, index=False)
                        segments_info.append((output_path, original_filename))
                        segment_count += 1
                        
                        logging.info(f"保留最後的部分片段: {remaining_frames} 幀")
                else:
                    logging.info(f"跳過最後的過短片段: 剩餘 {remaining_frames} 幀 (少於最小要求 {min_segment_length})")
                
                break

            current_end_frame_id = all_frames[current_end_frame_idx]
            segment_df = df.iloc[current_start_frame_idx : current_end_frame_idx + 1].copy()
            segment_filename = f"{base_name_without_ext}_seg{segment_count}_{current_start_frame_id}-{current_end_frame_id}.csv"
            output_path = os.path.join(output_segment_dir, segment_filename)
            segment_df.to_csv(output_path, index=False)
            segments_info.append((output_path, original_filename))

            next_start_frame_id = current_start_frame_id + segment_interval
            try:
                current_start_frame_idx = all_frames.index(next_start_frame_id)
            except ValueError:
                break

            segment_count += 1

        logging.info(f"Segmented {original_filename} into {segment_count} segments.")
        return segments_info

    except Exception as e:
        logging.error(f"Error segmenting {input_csv_path}: {str(e)}")
        return []


def load_keypoint_csv(file_path, target_frames=None, img_shape=(1080, 1920), selected_parts=None,
                     min_segment_length=None, keep_original_dimensions=True):
    try:
        df = pd.read_csv(file_path)
        effective_min_length = min_segment_length if min_segment_length is not None else (target_frames if target_frames is not None else 150)
        if len(df) < effective_min_length:
            logging.info(f"跳過 {file_path}: 只有 {len(df)} 幀 (少於最小要求 {effective_min_length})")
            return None

        all_parts = [
            'Nose', 'LEye', 'REye', 'LEar', 'REar',
            'LShoulder', 'RShoulder', 'LElbow', 'RElbow',
            'LWrist', 'RWrist', 'LHip', 'RHip',
            'LKnee', 'RKnee', 'LAnkle', 'RAnkle'
        ]

        if selected_parts is None:
            selected_parts = all_parts

        part_mask = [1 if part in selected_parts else 0 for part in all_parts]
        num_frames = len(df)
        keypoint_array = np.zeros((1, num_frames, 17, 2))
        keypoint_score = np.zeros((1, num_frames, 17))

        for kp_idx, mask in enumerate(part_mask):
            if mask:
                keypoint_score[0, :, kp_idx] = 1.0

        for frame_idx, row in df.iterrows():
            if frame_idx >= num_frames:
                break
            try:
                keypoints_list = ast.literal_eval(row['keypoints'])
                for kp_idx, kp in enumerate(keypoints_list):
                    if part_mask[kp_idx]:
                        keypoint_array[0, frame_idx, kp_idx, 0] = float(kp[0])
                        keypoint_array[0, frame_idx, kp_idx, 1] = float(kp[1])
            except Exception:
                for kp_idx, kp_name in enumerate(all_parts):
                    x_col = f"{kp_name}_x"
                    y_col = f"{kp_name}_y"
                    if x_col in row and y_col in row and part_mask[kp_idx]:
                        keypoint_array[0, frame_idx, kp_idx, 0] = float(row[x_col])
                        keypoint_array[0, frame_idx, kp_idx, 1] = float(row[y_col])

        return {
            'keypoint': keypoint_array,
            'keypoint_score': keypoint_score,
            'total_frames': num_frames,
            'img_shape': (img_shape[1], img_shape[0]),
            'original_shape': (img_shape[1], img_shape[0])
        }

    except Exception as e:
        logging.error(f"Error loading {file_path}: {str(e)}")
        return None

def main(keypoints_folder, labels_file_path, label_name, output_dir, img_shape=(1080, 1920),
         selected_parts=None, segment_length=None, segment_interval=None,
         min_segment_length=None, remove_partial_segments=False):
    os.makedirs(output_dir, exist_ok=True)
    segmented_keypoints_folder = os.path.join(output_dir, 'segments')
    os.makedirs(segmented_keypoints_folder, exist_ok=True)

    labels_df = pd.read_csv(labels_file_path)
    labels_dict = dict(zip(labels_df['keypoint_file'], labels_df[label_name]))
    id_dict = dict(zip(labels_df['keypoint_file'], labels_df['id']))

    original_keypoint_files = glob.glob(os.path.join(keypoints_folder, "*.csv"))
    logging.info(f"Found {len(original_keypoint_files)} original keypoint CSV files")

    all_processed_files_info = []
    for original_file_path in original_keypoint_files:
        original_filename = os.path.basename(original_file_path)
        if original_filename not in labels_dict:
            logging.info(f"Skipping {original_filename}: No label found")
            continue
        segments_info = segment_and_save_csvs(
            original_file_path, segmented_keypoints_folder, segment_length,
            segment_interval, min_segment_length, remove_partial_segments
        )
        all_processed_files_info.extend(segments_info)

    logging.info(f"Total processed CSV files (including segments): {len(all_processed_files_info)}")

    annotations = []
    file_source_map = {}
    for processed_file_path, original_filename in all_processed_files_info:
        processed_file_name = os.path.basename(processed_file_path)
        keypoint_data = load_keypoint_csv(processed_file_path, segment_length, img_shape, selected_parts, min_segment_length)
        if keypoint_data is None:
            continue

        file_id = processed_file_name.replace('.csv', '')
        source_id = id_dict.get(original_filename)
        label = labels_dict.get(original_filename)

        if source_id is None or label is None:
             logging.warning(f"Could not find source_id or label for {original_filename}. Skipping.")
             continue

        annotation = {
            'frame_dir': file_id,
            'source_id': source_id,
            'label': int(label),
            'total_frames': keypoint_data['total_frames'],
            'keypoint': keypoint_data['keypoint'],
        }
        annotations.append(annotation)
        file_source_map[file_id] = source_id

    sources = {source_id: [] for source_id in file_source_map.values()}
    for file_id, source_id in file_source_map.items():
        sources[source_id].append(file_id)

    source_ids = list(sources.keys())
    train_sources, val_sources = train_test_split(source_ids, test_size=0.2, random_state=44)
    train_videos = [vid for source in train_sources for vid in sources[source]]
    val_videos = [vid for source in val_sources for vid in sources[source]]

    annotation_map = {anno['frame_dir']: anno for anno in annotations}

    for phase, video_list in [('train', train_videos), ('val', val_videos)]:
        logging.info(f"Processing phase: {phase}...")

        sample_names = video_list
        sample_labels = [annotation_map[file_id]['label'] for file_id in video_list]

        label_path = os.path.join(output_dir, f'{phase}_label.pkl')
        with open(label_path, 'wb') as f:
            pickle.dump((sample_names, sample_labels), f)
        logging.info(f"Saved labels for {phase} to {label_path}")

        num_samples = len(video_list)
        max_frames = segment_length if segment_length is not None else 150
        num_joints = 17
        num_persons = 1

        data_array = np.zeros((num_samples, 3, max_frames, num_joints, num_persons), dtype=np.float32)

        for i, file_id in enumerate(tqdm(video_list, desc=f"Generating data for {phase}")):
            anno = annotation_map[file_id]
            keypoint_data = anno['keypoint']
            num_frames = anno['total_frames']
            
            data_c_first = np.zeros((3, num_frames, num_joints, num_persons))
            data_c_first[:2, :, :, :] = keypoint_data[0].transpose(2, 0, 1)[..., np.newaxis]
            
            data_array[i, :, :num_frames, :, :] = data_c_first

        logging.info(f"Saving data for {phase}...")

        data_path = os.path.join(output_dir, f'{phase}_data.npy')
        np.save(data_path, data_array)
        logging.info(f"Saved data for {phase} to {data_path}")

    # ===== 標籤分布統計並寫入 CSV =====
    all_labels = sorted(set([anno['label'] for anno in annotations]))
    train_clip_stats = {label: 0 for label in all_labels}
    val_clip_stats = {label: 0 for label in all_labels}
    train_source_stats = {label: 0 for label in all_labels}
    val_source_stats = {label: 0 for label in all_labels}

    # 計算 clips 標籤分布
    for file_id in train_videos:
        label = annotation_map[file_id]['label']
        train_clip_stats[label] += 1
    for file_id in val_videos:
        label = annotation_map[file_id]['label']
        val_clip_stats[label] += 1

    # 計算 sources 標籤分布
    for source_id in train_sources:
        # 取該 source 下所有 clip 的標籤（假設同一 source 標籤一致）
        labels = [annotation_map[file_id]['label'] for file_id in sources[source_id]]
        for label in set(labels):
            train_source_stats[label] += 1
    for source_id in val_sources:
        labels = [annotation_map[file_id]['label'] for file_id in sources[source_id]]
        for label in set(labels):
            val_source_stats[label] += 1

    # 總體統計
    total_clip_stats = {label: train_clip_stats[label] + val_clip_stats[label] for label in all_labels}
    total_source_stats = {label: train_source_stats[label] + val_source_stats[label] for label in all_labels}

    # 統計結果寫入 CSV
    stats_rows = []
    for label in all_labels:
        stats_rows.append({
            'label': label,
            'train_clips': train_clip_stats[label],
            'val_clips': val_clip_stats[label],
            'total_clips': total_clip_stats[label],
            'train_sources': train_source_stats[label],
            'val_sources': val_source_stats[label],
            'total_sources': total_source_stats[label]
        })
    stats_df = pd.DataFrame(stats_rows)
    print(stats_df)
    stats_csv_path = os.path.join(output_dir, f"{label_name}_stats.csv")
    stats_df.to_csv(stats_csv_path, index=False)
    logging.info(f"標籤統計已寫入 {stats_csv_path}")

    logging.info("Data generation complete.")


if __name__ == "__main__":
    import sys
    sys.path.append(os.path.abspath('../src'))

    from config import ConfigManager
    from utils.logger_config import setup_logging

    setup_logging()

    config_manager = ConfigManager("config.yaml")
    config = config_manager.config.get("data")

    keypoints_folder = "../outputs/csv"
    metadata_file = "../outputs/2025_normal_keypoint_analysis_metadata.csv"
    label_name = "result_months_range"
    output = "../outputs/pickle"
    
    segment_length = 300
    segment_interval = None
    min_segment_length = 90
    remove_partial_segments = False
    
    selected_parts_config = config.get("selected_parts", None)
    if selected_parts_config is not None:
        selected_parts = selected_parts_config
    else:
        # selected_parts = ['Nose', 'LHip', 'RHip', 'LKnee', 'RKnee', 'LAnkle', 'RAnkle']
        selected_parts = ['Nose', 'LShoulder', 'RShoulder', 'LElbow', 'RElbow',
            'LWrist', 'RWrist', 'LHip', 'RHip',
            'LKnee', 'RKnee', 'LAnkle', 'RAnkle']

    # if "min_segment_length" in config:
    #     min_segment_length = config.get("min_segment_length")
    if "remove_partial_segments" in config:
        remove_partial_segments = config.get("remove_partial_segments")

    main(keypoints_folder, metadata_file, label_name, output,
         img_shape=(1080, 1920),
         selected_parts=selected_parts,
         segment_length=segment_length,
         segment_interval=segment_interval,
         min_segment_length=min_segment_length,
         remove_partial_segments=remove_partial_segments)
