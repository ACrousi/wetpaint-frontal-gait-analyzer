import requests
import argparse
import sys
import os
import json

def main():
    parser = argparse.ArgumentParser(description="Test Video Processing API")
    parser.add_argument("--host", default="127.0.0.1", help="API Host")
    parser.add_argument("--port", default=8000, type=int, help="API Port")
    parser.add_argument("--video", default="data/raws/in.MOV", help="Path to video file")
    parser.add_argument("--case_id", default="test_case_001", help="Case ID")
    parser.add_argument("--months", default=12, type=int, help="Child's age in months")

    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}/process_video"
    
    # Ensure absolute path
    video_path = os.path.abspath(args.video)
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        # Try relative to project root if running from elsewhere? 
        # But we assume running from root.
        sys.exit(1)

    payload = {
        "case_id": args.case_id,
        "videopath": video_path,
        "months": args.months
    }

    print(f"Sending request to {url}...")
    print(f"Payload: {json.dumps(payload, indent=2, ensure_ascii=False)}")

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        print("\nResponse:")
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    except ImportError:
        print("\nError: 'requests' library not found. Please run 'pip install requests' to use this script.")
    except requests.exceptions.ConnectionError:
        print(f"\nError: Could not connect to API at {url}. Is the server running?")
        print("Run 'python start_api.py' in a separate terminal.")
    except Exception as e:
        print(f"\nError: {e}")
        if hasattr(e, 'response') and e.response is not None:
             print(f"Server response: {e.response.text}")

if __name__ == "__main__":
    main()
