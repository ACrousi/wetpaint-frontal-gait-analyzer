#!/usr/bin/env python3
"""
測試個案級彙整功能
"""

import sys
import os
from pathlib import Path

# 添加src到路徑
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from config.config import ConfigManager
from src.core.export_service import ExportService

def test_case_summary():
    """測試個案彙整功能"""
    print("開始測試個案彙整功能...")

    # 初始化配置
    config_path = 'config/config.yaml'
    if not Path(config_path).exists():
        print(f"配置檔案不存在: {config_path}")
        return

    config_manager = ConfigManager(config_path)
    config = config_manager.config

    # 初始化ExportService
    export_service = ExportService(config.get("export", {}))

    # 測試export_case_summary_csv
    case_id = "test_case_001"
    track_meta_csv = "outputs/csv/track_analysis_metadata.csv"
    out_csv = "outputs/csv/case_analysis_metadata.csv"

    if not Path(track_meta_csv).exists():
        print(f"軌跡metadata CSV不存在: {track_meta_csv}")
        print("請先運行影片處理以生成測試資料")
        return

    print(f"測試彙整 case_id: {case_id}")
    result_path = export_service.export_case_summary_csv(case_id, track_meta_csv, out_csv)

    if result_path:
        print(f"成功輸出個案彙整至: {result_path}")
        # 讀取並顯示結果
        import pandas as pd
        df = pd.read_csv(result_path, encoding='utf-8-sig')
        print("輸出內容:")
        print(df.head())
    else:
        print("個案彙整失敗")

if __name__ == "__main__":
    test_case_summary()