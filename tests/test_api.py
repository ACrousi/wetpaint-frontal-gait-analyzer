#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
測試影片處理 API
"""

import requests
import json
import sys
from pathlib import Path

def test_api():
    """測試 API 基本功能"""
    # API 基礎 URL
    base_url = "http://localhost:8000"
    
    # 測試健康檢查端點
    print("測試健康檢查端點...")
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            print("✓ 健康檢查通過")
            print(f"  響應: {response.json()}")
        else:
            print(f"✗ 健康檢查失敗，狀態碼: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ 無法連接到 API 服務，請確保服務正在運行")
        return False
    except Exception as e:
        print(f"✗ 健康檢查時發生錯誤: {e}")
        return False
    
    print("\nAPI 測試完成")
    return True

if __name__ == "__main__":
    test_api()