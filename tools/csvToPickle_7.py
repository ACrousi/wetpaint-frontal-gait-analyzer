import os
import glob
import numpy as np
import pandas as pd
import pickle
import ast
import logging # 導入 logging 模組
from sklearn.model_selection import train_test_split
import math # Needed for ceil

def segment_and_save_csvs(input_csv_path, output_segment_dir, segment_length, segment_interval=None):
    """
    Segments a single keypoint CSV file into smaller CSV files based on frame ranges.

    Args:
        input_csv_path (str): Path to the input keypoint CSV file.
        output_segment_dir (str): Directory to save the segmented CSV files.
        segment_length (int): The desired length of each segment in frames.
        segment_interval (int, optional): The interval between the start of segments.
                                          If None, it defaults to segment_length (no overlap).

    Returns:
        list: A list of tuples, where each tuple contains the path to a saved segment CSV
              and the original input CSV filename.
    """
    if segment_length is None:
        # If no segment_length is provided, just return the original file info
        original_filename = os.path.basename(input_csv_path)
        return [(input_csv_path, original_filename)]

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

        # If segment_length is greater than the total frames, don't segment
        if segment_length > total_frames_in_csv:
             logging.info(f"Skipping segmentation for {input_csv_path}: segment_length ({segment_length}) > total_frames ({total_frames_in_csv})")
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

            # Check if the segment goes beyond the end of the available frames
            if current_end_frame_idx >= total_frames_in_csv:
                # If the remaining frames are less than segment_length, we can either discard or take the last segment
                # Let's take the last segment if it's at least min_segment_length (though min_segment_length is not a param here, let's assume we need full length for now)
                # Or, simply stop if we can't form a full segment
                break # Stop if a full segment cannot be formed

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


def load_keypoint_csv(file_path, target_frames=None, img_shape=(1080, 1920)):
    """Load keypoint data from a CSV file."""
    try:
        df = pd.read_csv(file_path)

        # Check if file has at least target_frames if specified
        if target_frames is not None and len(df) < target_frames:
            logging.info(f"Skipping {file_path}: Only has {len(df)} frames (less than {target_frames})")
            return None

        # Initialize arrays
        num_frames = len(df)
        # Assuming the CSV columns are in the correct order for the 7 keypoints
        # Nose_x, Nose_y, LHip_x, LHip_y, RHip_x, RHip_y, LKnee_x, LKnee_y, RKnee_x, RKnee_y, LAnkle_x, LAnkle_y, RAnkle_x, RAnkle_y
        # The original code had 17 keypoints in the comment but only used 7 in the loop.
        # Let's stick to 7 keypoints as per the loop logic.
        keypoint_array = np.zeros((1, num_frames, 7, 2))  # [M=1, T=frames, V=7, C=2]
        keypoint_score = np.ones((1, num_frames, 7))      # Default confidence 1.0

        # Process each frame
        for frame_idx, row in df.iterrows():
            if frame_idx >= num_frames:
                break

            # Parse the keypoints string into a list of coordinates
            try:
                keypoints_str = row['keypoints']
                keypoints_list = ast.literal_eval(keypoints_str)

                # Fill the keypoint array
                for kp_idx, kp in enumerate(keypoints_list):
                    if kp_idx < 7:  # Ensure within bounds
                        keypoint_array[0, frame_idx, kp_idx, 0] = float(kp[0])  # x coordinate
                        keypoint_array[0, frame_idx, kp_idx, 1] = float(kp[1])  # y coordinate
            except Exception:
                # Use individual columns as fallback
                for kp_idx, kp_name in enumerate([
                    'Nose', 'LHip', 'RHip',
                    'LKnee', 'RKnee', 'LAnkle', 'RAnkle'
                ]):
                    x_col = f"{kp_name}_x"
                    y_col = f"{kp_name}_y"

                    if x_col in row and y_col in row:
                        x_val = float(row[x_col])
                        y_val = float(row[y_col])
                        keypoint_array[0, frame_idx, kp_idx, 0] = x_val
                        keypoint_array[0, frame_idx, kp_idx, 1] = y_val

        return {
            'keypoint': keypoint_array,
            'keypoint_score': keypoint_score,
            'total_frames': num_frames,
            'img_shape': (img_shape[1], img_shape[0]),  # (height, width)
            'original_shape': (img_shape[1], img_shape[0]),  # (height, width)
        }

    except Exception as e:
        logging.error(f"Error loading {file_path}: {str(e)}")
        return None

def main(keypoints_folder, labels_file_path, label_name, output_dir, target_frames=None, img_shape=(1080, 1920), segment_length=None, segment_interval=None):
    """
    Processes keypoint CSV files, segments them if specified, and creates a COCO format pickle dataset.

    Args:
        keypoints_folder (str): Folder containing the input keypoint CSV files.
        labels_file_path (str): Path to the CSV file with video labels.
        label_name (str): The name of the column in the labels file to use as the label.
        output_dir (str): Directory to save the output pickle file and segmented CSVs.
        target_frames (int, optional): Minimum number of frames required for a CSV to be processed.
                                       If None, all CSVs are processed.
        img_shape (tuple, optional): The shape of the images (width, height). Defaults to (1080, 1920).
        segment_length (int, optional): The desired length of each segment in frames.
                                        If None, no segmentation is performed.
        segment_interval (int, optional): The interval between the start of segments.
                                          If None and segment_length is provided, it defaults to segment_length.
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

        # Segment the original file and get info for all resulting segment files
        segments_info = segment_and_save_csvs(
            original_file_path,
            segmented_keypoints_folder,
            segment_length,
            segment_interval
        )
        all_processed_files_info.extend(segments_info)

    logging.info(f"Total processed CSV files (including segments): {len(all_processed_files_info)}")

    # Process files and create annotations
    annotations = []
    file_source_map = {}  # Map to track which source each file belongs to

    for processed_file_path, original_filename in all_processed_files_info:
        processed_file_name = os.path.basename(processed_file_path)

        # Load keypoint data from the processed file (segment or original)
        logging.info(f"Loading data from {processed_file_name}...")
        keypoint_data = load_keypoint_csv(processed_file_path, target_frames, img_shape)
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

    # Save as pickle file
    with open(output_path, 'wb') as f:
        pickle.dump(combined_dict, f)

    logging.info(f"Created COCO format dataset:")
    logging.info(f"  - Combined: {len(annotations)} videos ({len(train_videos)} train, {len(val_videos)} val)")
    logging.info(f"  - Sources: {len(source_ids)} total ({len(train_sources)} in train, {len(val_sources)} in val)")
    logging.info(f"Saved to {output_path}")

if __name__ == "__main__":
    # import argparse

    # parser = argparse.ArgumentParser(description='Convert keypoint CSVs to COCO format')
    # parser.add_argument('--keypoints_folder', required=True, help='Folder with keypoint CSVs')
    # parser.add_argument('--labels_file', required=True, help='CSV file with video labels')
    # parser.add_argument('--output', default='action_dataset.pkl', help='Output pickle file')
    # parser.add_argument('--segment_length', type=int, default=None, help='Length of each segment in frames')
    # parser.add_argument('--segment_interval', type=int, default=None, help='Interval between segment starts in frames')

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
    metadata_file = "../outputs/2024_result.csv"
    label_name = "result_months_range"
    output = "../outputs/pickle"

    # Read segment parameters from config
    segment_length = config.get("segment_length")
    segment_interval = config.get("segment_interval")
    target_frames = config.get("target_frames", None) # Also read target_frames if it exists

    main(keypoints_folder, metadata_file, label_name, output,
         target_frames=target_frames, # Pass target_frames
         segment_length=segment_length,
         segment_interval=segment_interval)
