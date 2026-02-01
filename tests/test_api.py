import requests
import argparse
import sys
import os
import json
import time


def test_single_case(url: str, case_id: str, video_path: str, months: int) -> dict:
    """測試單一 case 並回傳結果"""
    
    if not os.path.exists(video_path):
        return {
            "case_id": case_id,
            "status": "error",
            "error": f"Video file not found: {video_path}"
        }

    payload = {
        "case_id": case_id,
        "videopath": video_path,
        "months": months
    }

    print(f"\n{'='*60}")
    print(f"Testing Case: {case_id}")
    print(f"Video: {video_path}")
    print(f"{'='*60}")

    try:
        start_time = time.time()
        response = requests.post(url, json=payload)
        elapsed = time.time() - start_time
        
        response.raise_for_status()
        result = response.json()
        result["elapsed_seconds"] = round(elapsed, 2)
        
        print(f"✓ 成功 (耗時: {elapsed:.2f}s)")
        if "predicted_age" in result:
            print(f"  預測月齡: {result['predicted_age']:.2f}")
        
        return {
            "case_id": case_id,
            "status": "success",
            "result": result
        }
        
    except requests.exceptions.ConnectionError:
        return {
            "case_id": case_id,
            "status": "error",
            "error": f"Could not connect to API at {url}"
        }
    except Exception as e:
        error_msg = str(e)
        if hasattr(e, 'response') and e.response is not None:
            error_msg += f" - {e.response.text}"
        return {
            "case_id": case_id,
            "status": "error",
            "error": error_msg
        }


def main():
    parser = argparse.ArgumentParser(description="Test Video Processing API with Multiple Cases")
    parser.add_argument("--host", default="127.0.0.1", help="API Host")
    parser.add_argument("--port", default=8000, type=int, help="API Port")
    parser.add_argument("--video_dir", default="data/raws", help="Directory containing video files")
    parser.add_argument("--months", default=18, type=int, help="Child's age in months (default)")

    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}/process_video"
    video_dir = os.path.abspath(args.video_dir)

    # 定義三個測試 case
    test_cases = [
        {"case_id": "test_case_in", "video": "in.MOV", "months": args.months},
        {"case_id": "test_case_in2", "video": "in2.MOV", "months": args.months},
        {"case_id": "test_case_in3", "video": "in3.MOV", "months": args.months},
    ]

    print(f"API URL: {url}")
    print(f"Video Directory: {video_dir}")
    print(f"Testing {len(test_cases)} cases...")

    results = []
    
    for case in test_cases:
        video_path = os.path.join(video_dir, case["video"])
        result = test_single_case(
            url=url,
            case_id=case["case_id"],
            video_path=video_path,
            months=case["months"]
        )
        results.append(result)

    # 輸出總結
    print(f"\n{'='*60}")
    print("測試結果總結")
    print(f"{'='*60}")
    
    success_count = sum(1 for r in results if r["status"] == "success")
    print(f"成功: {success_count}/{len(results)}")
    
    for r in results:
        status_icon = "✓" if r["status"] == "success" else "✗"
        print(f"  {status_icon} {r['case_id']}: {r['status']}")
        if r["status"] == "error":
            print(f"      Error: {r['error']}")
        elif "result" in r and "predicted_age" in r["result"]:
            print(f"      預測月齡: {r['result']['predicted_age']:.2f}")

    # 輸出完整 JSON 結果
    print(f"\n{'='*60}")
    print("完整 JSON 結果:")
    print(f"{'='*60}")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
