import os
import re
import pandas as pd
from datetime import datetime
from config.config import ConfigManager
from src.utils.videoPreprocessing import get_video_paths

def extract_info(text):
    # 匹配ID（4個英文字母後跟數字）
    id_match = re.search(r'([A-Z]{4}\d+)', text)
    id_value = id_match.group(1) if id_match else None

    # 匹配姓名（假設姓名是由2到4個中文字組成）
    name_match = re.search(r'([\u4e00-\u9fa5]{2,4})', text)
    name = name_match.group(1) if name_match else None

    # 匹配年齡並轉換為月齡
    age_matches = re.findall(r'(\d+)([yYmM])', text)
    age_months = 0
    for value, unit in age_matches:
        if unit.lower() == 'y':
            age_months += int(value) * 12
        else:
            age_months += int(value)

    return id_value, name, age_months if age_months > 0 else None

def extract_date_from_video_name(video_name):
    """
    從 video_name 中提取日期
    例如: "20240717_015058000_iOS.MOV" -> "20240717"
    """
    # 匹配 YYYYMMDD 格式
    date_match = re.search(r'(\d{4})(\d{2})(\d{2})', video_name)
    if date_match:
        year = date_match.group(1)
        month = date_match.group(2)
        day = date_match.group(3)
        return f"{year}{month}{day}"
    return None

def main(output_filename='2024.csv'):
    # with open('./src/config/config.yaml', 'r') as f:
    #     config = yaml.safe_load(f)
    config_manager = ConfigManager("./config.yaml")
    config = config_manager.config

    data_config = config.get("data")
    input_path = data_config.get("input_path", "")
    output_base = data_config.get("output_path", "")

    # Initialize empty list for video metadata
    video_metadata_list = []

    print(input_path)
    if os.path.isdir(input_path):
        original_videos_path = get_video_paths(input_path)
        for video_path in original_videos_path:
            filename = os.path.basename(video_path)
            parent = os.path.basename(os.path.dirname(video_path))
            grandparent = os.path.basename(os.path.dirname(os.path.dirname(video_path)))
            id_val, name, age_months = extract_info(parent)
            
            # 從 video_name 中提取日期
            video_date = extract_date_from_video_name(filename)
            
            # Collect video metadata
            video_metadata_list.append({
                'original_video': video_path,
                'video_name': filename,
                'child_info': parent,
                'id': id_val,
                'name': name,
                'age_months': age_months,
                'dir_name': grandparent,
                'video_date': video_date
            })
            # print(f"Extracted metadata for {video_path}") # Optional: more verbose logging

        # Create DataFrame from extracted video metadata, ensuring unique video_name
        video_meta_df = pd.DataFrame(video_metadata_list)
        if not video_meta_df.empty:
             # Keep the first occurrence if duplicates exist based on video_name
            video_meta_df = video_meta_df.drop_duplicates(subset=['video_name'], keep='first')
            print(f"Created initial DataFrame with {len(video_meta_df)} unique video entries.")
        else:
            print("No video metadata extracted.")
            return # Exit if no videos found

        # --- Merging Logic ---
        merged_df = pd.DataFrame() # Initialize final DataFrame

        # 1. Check for seg_label (video_seg.csv)
        seg_label_path = data_config.get('seg_label_path')
        if seg_label_path and os.path.exists(seg_label_path):
            # try:
            seg_df = pd.read_csv(seg_label_path)
            print(f"Read {len(seg_df)} records from video segments file: {seg_label_path}")
            # Merge video metadata INTO seg_df (seg_df is the primary table)
            merged_df = pd.merge(seg_df, video_meta_df, on='video_name', how='left')
            print(f"Merged video metadata into segments. Result has {len(merged_df)} rows.")
            # except Exception as e:
            #     print(f"Error reading or merging video segments: {e}")
            #     # Fallback: use video_meta_df if seg merge fails but seg file exists
            #     merged_df = video_meta_df 
        else:
            # If no seg_label file, the base is the video metadata
            print("No video segments file specified or found. Using extracted video metadata as base.")
            merged_df = video_meta_df

        # 2. Merge labels if label_path exists
        label_path = data_config.get('label_path')
        if label_path and os.path.exists(label_path) and not merged_df.empty:
            # try:
            labels_df = pd.read_csv(label_path)
            print(f"Read {len(labels_df)} records from labels file: {label_path}")
            cols_to_use = ['id'] + [f for f in ['diagnosis_date', 'result_months',
                                'developmental_retardation', 'result_months_range', 'age_months_range']
                            if f in labels_df.columns]
            labels_df = labels_df[cols_to_use]
            
            if 'id' in merged_df.columns and 'video_date' in merged_df.columns:
                # 執行自定義合併邏輯：同時匹配 id 和 diagnosis_date (與 video_date 比較)
                merged_rows = []
                
                for _, video_row in merged_df.iterrows():
                    video_id = video_row['id']
                    video_date = video_row['video_date']
                    
                    if pd.isna(video_id) or pd.isna(video_date):
                        # 如果 ID 或日期為空，保留原始記錄但不合併 label
                        merged_rows.append(video_row.to_dict())
                        continue
                    
                    # 找到匹配的 label：同樣的 id 且 diagnosis_date 與 video_date 一致
                    matching_labels = labels_df[
                        (labels_df['id'] == video_id) &
                        (labels_df['diagnosis_date'] == video_date)
                    ]
                    
                    if not matching_labels.empty:
                        # 如果找到匹配的 label，使用第一個匹配結果
                        label_row = matching_labels.iloc[0]
                        merged_row = video_row.to_dict()
                        # 添加 label 欄位
                        for col in labels_df.columns:
                            if col != 'id':  # 避免重複 id 欄位
                                merged_row[col] = label_row[col]
                        merged_rows.append(merged_row)
                    else:
                        # 如果沒有找到匹配的 label，保留原始記錄
                        merged_rows.append(video_row.to_dict())
                        print(f"✗ 未找到匹配: ID={video_id}, date={video_date}")
                
                # 重建 DataFrame
                merged_df = pd.DataFrame(merged_rows)
                print(f"完成自定義合併。結果有 {len(merged_df)} 行。")
            else:
                print("跳過 label 合併：'id' 或 'video_date' 欄位未找到。")
            # except Exception as e:
            #     print(f"Error reading or merging labels: {e}")
        else:
             print("No label file specified or found, or base DataFrame is empty. Skipping label merge.")

        # --- Save Final DataFrame ---
        if not merged_df.empty:
            output_path = os.path.join(output_base, output_filename)
            
            # Handle existing file
            if os.path.exists(output_path):
                base, ext = os.path.splitext(output_filename)
                output_path = os.path.join(output_base, f"{base}_out{ext}")
            
            # Sort before saving (optional, but good practice)
            if 'original_video' in merged_df.columns:
                 merged_df = merged_df.sort_values(by='original_video')
            elif 'video_name' in merged_df.columns:
                 merged_df = merged_df.sort_values(by='video_name')

            merged_df.to_csv(output_path, encoding='utf-8-sig', index=False)
            print(f"Final metadata saved to {output_path}")
        else:
            print("Resulting DataFrame is empty. No file saved.")
    return



import sys
sys.path.append(os.path.join(os.path.dirname(__file__), './src'))
# sys.path.append(os.path.join(os.path.dirname(__file__), './vendor/BoTSORT'))

if __name__ == "__main__":
    import sys
    output_filename = '2024_ddd.csv'  # 預設輸出檔案名稱
    if len(sys.argv) > 1:
        output_filename = sys.argv[1]  # 如果提供了命令列參數，則使用該參數作為輸出檔案名稱
    main(output_filename)
