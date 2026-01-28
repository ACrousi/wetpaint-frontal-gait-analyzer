#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FastAPI 影片處理服務啟動腳本
"""

import uvicorn
import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="啟動影片處理 API 服務")
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="主機地址 (默認: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="端口號 (默認: 8000)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="開發模式下啟用自動重載"
    )
    
    args = parser.parse_args()
    
    # 確保在正確的目錄中運行
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))
    
    print(f"正在啟動影片處理 API 服務...")
    print(f"訪問地址: http://{args.host}:{args.port}")
    print(f"API 文檔: http://{args.host}:{args.port}/docs")
    
    uvicorn.run(
        "src.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )

if __name__ == "__main__":
    main()