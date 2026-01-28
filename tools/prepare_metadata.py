import os
import re
import pandas as pd
from datetime import datetime, timedelta
from glob import glob


def extract_info(text):
    # 匹配ID（4個英文字母後跟數字）
    id_match = re.search(r'([A-Z]{4}\d+)', text)
    id_value = id_match.group(1) if id_match else None

    # 匹配年齡並轉換為月齡
    age_matches = re.findall(r'(\d+)([yYmM])', text)
    age_months = 0
    for value, unit in age_matches:
        if unit.lower() == 'y':
            age_months += int(value) * 12
        else:
            age_months += int(value)

    return id_value, age_months if age_months > 0 else None


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


def get_video_paths(base_path):
    """
    掃描指定路徑及其子資料夾中的所有影片檔案
    支援的影片格式: .mp4, .mov, .avi, .mkv
    """
    video_extensions = ['*.mp4', '*.MP4', '*.mov', '*.MOV', '*.avi', '*.AVI', '*.mkv', '*.MKV']
    video_paths = []
    
    for ext in video_extensions:
        # 掃描當前目錄
        video_paths.extend(glob(os.path.join(base_path, ext)))
        # 掃描所有子目錄
        video_paths.extend(glob(os.path.join(base_path, '**', ext), recursive=True))
    
    # 去除重複路徑
    video_paths = list(set(video_paths))
    return video_paths


def parse_label_value(value):
    """
    解析粗大動作欄位的值，返回對應的數字標籤
    "發展正常" -> 0
    "邊緣遲緩" -> 1
    "發展遲緩" -> 2
    """
    if pd.isna(value):
        return None
    
    value_str = str(value)
    if "發展正常" in value_str:
        return 0
    elif "邊緣遲緩" in value_str:
        return 1
    elif "發展遲緩" in value_str:
        return 2
    
    return None


def parse_date_string(date_str):
    """
    將日期字串解析為 datetime 物件
    支援格式: YYYYMMDD, YYYY/M/D, YYYY-M-D
    """
    if pd.isna(date_str):
        return None
    
    date_str = str(date_str).strip()
    
    # 嘗試 YYYYMMDD 格式
    if re.match(r'^\d{8}$', date_str):
        try:
            return datetime.strptime(date_str, '%Y%m%d')
        except ValueError:
            pass
    
    # 嘗試其他格式
    for fmt in ['%Y/%m/%d', '%Y-%m-%d', '%Y/%m/%d', '%Y-%m-%d']:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    # 嘗試 datetime 物件
    if hasattr(date_str, 'strftime'):
        return date_str
    
    return None


def calculate_age_months(eval_date, birthday):
    """
    計算評估日期與生日之間的月齡差（四捨五入到整數）
    
    參數:
        eval_date: 評估日期 (datetime 物件)
        birthday: 生日 (datetime 物件)
    
    返回:
        月齡（四捨五入的整數），若無法計算則返回 None
    """
    if eval_date is None or birthday is None:
        return None
    
    try:
        # 計算總天數差
        days_diff = (eval_date - birthday).days
        # 轉換為月份（一個月約 30.44 天），四捨五入
        months = round(days_diff / 30.44)
        return int(months) if months >= 0 else None
    except Exception:
        return None


def load_and_process_label_csv(label_csv_path):
    """
    載入並處理標籤 CSV 檔案
    1. 去除空白列
    2. 解析 "受測者編號"、"評估日期"、"生日"
    3. 計算月齡（評估日期 - 生日）
    4. 解析 "粗大動作" 欄位
    
    返回包含 subject_id, eval_date (datetime), birthday (datetime), age_months, development_result 的 DataFrame
    """
    df = pd.read_csv(label_csv_path)
    
    # 去除所有空白列 (所有欄位都是空值的列)
    df = df.dropna(how='all')
    
    # 確保必要欄位存在
    required_columns = ['受測者編號', '評估日期', '粗大動作']
    for col in required_columns:
        if col not in df.columns:
            print(f"警告: 標籤 CSV 缺少必要欄位 '{col}'")
            return None
    
    # 檢查生日欄位是否存在
    has_birthday = '生日' in df.columns
    if not has_birthday:
        print("提示: 標籤 CSV 沒有 '生日' 欄位，將使用資料夾名稱中的月齡")
    
    # 提取受測者編號
    df['subject_id'] = df['受測者編號'].apply(lambda x: str(x).strip() if pd.notna(x) else None)
    
    # 解析評估日期為 datetime 物件
    df['eval_date'] = df['評估日期'].apply(parse_date_string)
    
    # 解析生日並計算月齡
    if has_birthday:
        df['birthday'] = df['生日'].apply(parse_date_string)
        df['age_months'] = df.apply(
            lambda row: calculate_age_months(row['eval_date'], row['birthday']),
            axis=1
        )
    else:
        df['birthday'] = None
        df['age_months'] = None
    
    # 解析粗大動作欄位
    df['development_result'] = df['粗大動作'].apply(parse_label_value)
    
    # 只保留有效的記錄 (subject_id 和 eval_date 必須有值)
    df = df.dropna(subset=['subject_id', 'eval_date'])
    
    return df[['subject_id', 'eval_date', 'age_months', 'development_result']]


def find_matching_label(video_subject_id, video_date, label_df, tolerance_days=10):
    """
    根據受測者編號和日期 (±tolerance_days) 尋找對應的標籤和月齡
    
    參數:
        video_subject_id: 影片的受測者編號
        video_date: 影片的日期 (datetime 物件)
        label_df: 標籤 DataFrame
        tolerance_days: 日期容差天數
    
    返回:
        (development_result, age_months) 元組，若無匹配則返回 (None, None)
    """
    if video_subject_id is None or video_date is None:
        return None, None
    
    # 篩選相同受測者編號的記錄
    subject_matches = label_df[label_df['subject_id'] == video_subject_id]
    
    if subject_matches.empty:
        return None, None
    
    # 計算日期差異並找到在容差範圍內的記錄
    best_match_result = None
    best_match_age = None
    min_diff = timedelta(days=tolerance_days + 1)
    
    for _, row in subject_matches.iterrows():
        if pd.notna(row['eval_date']):
            diff = abs(video_date - row['eval_date'])
            if diff <= timedelta(days=tolerance_days) and diff < min_diff:
                min_diff = diff
                best_match_result = row['development_result']
                best_match_age = row['age_months']
    
    return best_match_result, best_match_age


def main(videos_path, label_csv_path=None, output_filename='output_metadata.csv'):
    """
    主函數
    
    參數:
        videos_path: 影片存放路徑，會掃描該路徑及子資料夾中的所有影片
        label_csv_path: 標籤 CSV 檔案路徑 (可選)
        output_filename: 輸出檔案名稱
    """
    # Initialize empty list for video metadata
    video_metadata_list = []

    print(f"掃描影片路徑: {videos_path}")
    
    # 預先載入標籤 CSV (如果提供)
    label_df = None
    if label_csv_path and os.path.exists(label_csv_path):
        print(f"載入標籤檔案: {label_csv_path}")
        label_df = load_and_process_label_csv(label_csv_path)
        if label_df is not None:
            print(f"標籤檔案包含 {len(label_df)} 筆有效記錄")
    
    if os.path.isdir(videos_path):
        original_videos_path = get_video_paths(videos_path)
        print(f"找到 {len(original_videos_path)} 個影片檔案")
        
        for video_path in original_videos_path:
            filename = os.path.basename(video_path)
            parent = os.path.basename(os.path.dirname(video_path))
            id_val, age_months = extract_info(parent)
            
            # 從 video_name 中提取日期 (YYYYMMDD 格式)
            video_date_str = extract_date_from_video_name(filename)
            video_date = parse_date_string(video_date_str) if video_date_str else None
            
            # 建立 ID: id_val_YYYYMMDD (使用資料夾影片的日期)
            if id_val and video_date_str:
                combined_id = f"{id_val}_{video_date_str}"
            else:
                combined_id = id_val  # 如果沒有日期，只使用 id_val
            
            # 尋找對應的標籤和月齡 (使用 ±10 天容差)
            development_result = None
            label_age_months = None
            if label_df is not None and id_val and video_date:
                development_result, label_age_months = find_matching_label(id_val, video_date, label_df, tolerance_days=10)
            
            # 月齡優先使用標籤 CSV 計算的值 (評估日期 - 生日)，若無則使用資料夾名稱提取的值
            final_age_months = label_age_months if label_age_months is not None else age_months
            
            # Collect video metadata
            video_metadata_list.append({
                'original_video': video_path,
                'id': combined_id,
                'age_months': final_age_months,
                'development_result': development_result,
            })

        # Create DataFrame from extracted video metadata
        video_meta_df = pd.DataFrame(video_metadata_list)
        
        if not video_meta_df.empty:
            # 以 original_video 路徑為主鍵，保留所有影片記錄
            print(f"建立初始 DataFrame，包含 {len(video_meta_df)} 筆影片記錄。")
            
            if label_df is not None:
                matched_count = video_meta_df['development_result'].notna().sum()
                print(f"合併完成，{matched_count} 筆記錄有對應標籤。")
            
            # 儲存結果 (保留所有記錄，包括沒有對應標籤的)
            video_meta_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
            print(f"結果已儲存至: {output_filename}")
            
        else:
            print("未提取到任何影片元資料。")
            return

    else:
        print(f"錯誤: 路徑不存在或不是目錄: {videos_path}")

    return


if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='準備影片元資料')
    parser.add_argument('videos_path', type=str, help='影片存放路徑')
    parser.add_argument('--label_csv', type=str, default=None, help='標籤 CSV 檔案路徑')
    parser.add_argument('--output', type=str, default='2026_label.csv', help='輸出檔案名稱')
    
    args = parser.parse_args()
    
    main(args.videos_path, args.label_csv, args.output)
