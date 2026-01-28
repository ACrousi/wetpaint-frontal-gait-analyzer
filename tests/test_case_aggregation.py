#!/usr/bin/env python3
"""
最小化單元測試：測試 per-case 多影片 SegmentSummary 聚合功能
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# 添加 src 到路徑
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from core.analysis_service import AnalysisService
from core.export_service import ExportService
from pose_extract.track_solution.analysis.analysis_results import AnalysisResult
from pose_extract.track_solution.analysis.base import SegmentType

def create_mock_analysis_result(track_id: int, segment_type: str = "combined", include_list_metric: bool = False) -> AnalysisResult:
    """創建模擬的 AnalysisResult"""
    result = AnalysisResult(track_id)

    # 模擬 metrics DataFrame
    mock_data = {
        'body_proportion_head_ratio': [4.1, 4.2, 4.0, 4.3],
        'body_proportion_sitting_height_index': [55.0, 56.0, 54.5, 57.0],
        'torso_length': [0.25, 0.26, 0.24, 0.27],
        'hip_width': [0.15, 0.16, 0.14, 0.17]
    }
    df = pd.DataFrame(mock_data)
    result.add_metric("body_proportion", df)
    result.add_metric("torso_proportion", df)

    # 如果需要，添加 list 型指標（模擬某些指標如 step_time_list）
    if include_list_metric:
        list_df = pd.DataFrame({'step_time_list': [[0.8, 0.9, 1.0, 1.1]]})
        result.add_metric("step_time_metric", list_df)

    # 模擬 conditions (布林值)
    conditions_df = pd.DataFrame({'combined': [True, True, True, True]})
    result.add_condition(segment_type, conditions_df)

    return result

def test_aggregate_case_segment_summary():
    """測試 aggregate_case_segment_summary 方法"""
    print("測試 aggregate_case_segment_summary...")

    # 創建 AnalysisService
    config = {"case_aggregation": {"segment_type_filter": "combined", "exclude_metrics": []}}
    service = AnalysisService(config)

    # 模擬多影片的分析結果
    results_per_video = [
        {1: create_mock_analysis_result(1), 2: create_mock_analysis_result(2)},  # 第一支影片
        {3: create_mock_analysis_result(3)},  # 第二支影片
    ]

    # 測試聚合
    summary_df = service.aggregate_case_segment_summary(
        results_per_video,
        segment_type_filter="combined",
        exclude_metrics=[]
    )

    print(f"聚合結果 shape: {summary_df.shape}")
    print("聚合結果欄位:", list(summary_df.columns))
    print("聚合結果:")
    print(summary_df)

    # 驗證結果
    assert not summary_df.empty, "聚合結果不應為空"
    assert summary_df.shape[0] == 1, "應為單列結果"

    # 檢查統計欄位
    expected_cols = [
        'body_proportion_head_ratio_median', 'body_proportion_head_ratio_mean',
        'body_proportion_head_ratio_max', 'body_proportion_head_ratio_min',
        'body_proportion_head_ratio_std', 'body_proportion_head_ratio_cv'
    ]
    for col in expected_cols:
        assert col in summary_df.columns, f"缺少欄位: {col}"

    print("✓ aggregate_case_segment_summary 測試通過")

def test_aggregate_with_list_metrics():
    """測試包含 list 型指標的聚合"""
    print("\n測試包含 list 型指標的聚合...")

    config = {"case_aggregation": {"segment_type_filter": "combined", "exclude_metrics": []}}
    service = AnalysisService(config)

    # 創建包含 list 指標的結果
    results_per_video = [
        {1: create_mock_analysis_result(1, include_list_metric=True)},
        {2: create_mock_analysis_result(2, include_list_metric=True)},
    ]

    summary_df = service.aggregate_case_segment_summary(
        results_per_video,
        segment_type_filter="combined",
        exclude_metrics=[]
    )

    print(f"包含 list 指標的聚合結果 shape: {summary_df.shape}")
    print("包含 list 指標的聚合結果欄位:", list(summary_df.columns))

    # 檢查 list 型指標的統計欄位
    list_cols = [col for col in summary_df.columns if 'step_time_list' in col]
    assert len(list_cols) > 0, "應包含 list 指標的統計欄位"

    print("✓ list 型指標聚合測試通過")

def test_export_case_segment_summary():
    """測試 export_case_segment_summary 方法"""
    print("\n測試 export_case_segment_summary...")

    # 創建 ExportService
    config = {"export": {}}
    service = ExportService(config)

    # 創建測試 DataFrame
    test_df = pd.DataFrame({
        'test_metric_median': [1.5],
        'test_metric_mean': [1.6],
        'test_metric_std': [0.1]
    })

    # 測試導出
    output_path = service.export_case_segment_summary("test_case", test_df, "test_outputs")

    assert output_path is not None, "應返回輸出路徑"
    assert Path(output_path).exists(), f"檔案應存在: {output_path}"

    # 驗證檔案內容
    loaded_df = pd.read_csv(output_path)
    pd.testing.assert_frame_equal(test_df, loaded_df, check_dtype=False)

    print(f"✓ export_case_segment_summary 測試通過，輸出檔案: {output_path}")

    # 清理測試檔案
    Path(output_path).unlink(missing_ok=True)
    Path("test_outputs").rmdir()

def test_no_conditions_fallback():
    """測試無條件時的 fallback 行為"""
    print("\n測試無條件 fallback...")

    config = {"case_aggregation": {"segment_type_filter": "combined", "exclude_metrics": []}}
    service = AnalysisService(config)

    # 創建沒有條件的 AnalysisResult
    result = AnalysisResult(1)
    mock_data = {'metric_a': [1.0, 2.0, 3.0]}
    df = pd.DataFrame(mock_data)
    result.add_metric("test_metric", df)
    # 不添加條件，測試 fallback

    results_per_video = [{1: result}]

    summary_df = service.aggregate_case_segment_summary(
        results_per_video,
        segment_type_filter="nonexistent",  # 不存在的條件
        exclude_metrics=[]
    )

    assert not summary_df.empty, "即使無條件也應有結果"
    print("✓ 無條件 fallback 測試通過")

if __name__ == "__main__":
    print("開始測試 per-case 多影片 SegmentSummary 聚合功能...")

    try:
        test_aggregate_case_segment_summary()
        test_aggregate_with_list_metrics()
        test_export_case_segment_summary()
        test_no_conditions_fallback()
        print("\n🎉 所有測試通過！")
    except Exception as e:
        print(f"\n❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)