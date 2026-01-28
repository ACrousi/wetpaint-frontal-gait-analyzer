"""
測試 PyAV 讀取影片 rotation metadata 的腳本
用法: python test_rotation.py <video_path>
"""
import sys
import av
import numpy as np
import subprocess
import json
import logging


def get_rotation_via_ffprobe(file_path):
    """
    透過呼叫系統 ffprobe 指令獲取影片旋轉角度。
    這通常比 PyAV/OpenCV 更能識別各種怪異的 metadata 格式。
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream_tags=rotate:stream=width,height",  # 只抓 rotate tag 和長寬
        "-of", "json",
        file_path
    ]

    try:
        # 執行指令並獲取輸出
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        data = json.loads(result.stdout)
        
        # 解析 JSON
        streams = data.get("streams", [])
        if not streams:
            return 0, None
            
        stream = streams[0]
        tags = stream.get("tags", {})
        width = stream.get("width")
        height = stream.get("height")
        
        # 1. 嘗試直接讀取 rotate tag (最常見於 iOS/Android)
        if "rotate" in tags:
            return int(tags["rotate"]), {"width": width, "height": height, "tags": tags}
            
        # 2. 如果沒有 rotate tag，有些檔案會透過 side_data 顯示 (ffprobe json 有時會自動轉譯)
        # 但通常 ffprobe -show_entries stream_tags=rotate 已經足夠抓出 99% 的情況
        
        return 0, {"width": width, "height": height, "tags": tags}

    except Exception as e:
        logging.error(f"FFprobe 偵測失敗: {e}")
        return 0, None


def test_rotation(video_path: str):
    """測試讀取影片的 rotation metadata"""
    print(f"\n{'='*60}")
    print(f"測試影片: {video_path}")
    print(f"{'='*60}")
    
    try:
        container = av.open(video_path)
        stream = container.streams.video[0]
        
        # 基本影片資訊
        print(f"\n[基本資訊]")
        print(f"  寬度: {stream.width}")
        print(f"  高度: {stream.height}")
        print(f"  FPS: {float(stream.average_rate) if stream.average_rate else 'N/A'}")
        print(f"  總幀數: {stream.frames}")
        print(f"  編碼: {stream.codec.name}")
        
        # 印出 stream 的所有屬性
        print(f"\n[Stream 所有屬性]")
        stream_attrs = [attr for attr in dir(stream) if not attr.startswith('_')]
        print(f"  共 {len(stream_attrs)} 個屬性:")
        for attr in stream_attrs:
            try:
                value = getattr(stream, attr)
                if not callable(value):
                    print(f"    {attr}: {value}")
            except Exception as e:
                print(f"    {attr}: <error: {e}>")
        
        # 方法 1: 從 stream.metadata 取得 rotation
        print(f"\n[方法 1: stream.metadata]")
        print(f"  stream.metadata = {dict(stream.metadata)}")
        rotation_from_stream = stream.metadata.get('rotate', 'N/A')
        print(f"  rotation = {rotation_from_stream}")
        
        # 方法 2: 從 container.metadata 取得
        print(f"\n[方法 2: container.metadata]")
        print(f"  container.metadata = {dict(container.metadata)}")
        rotation_from_container = container.metadata.get('rotate', 'N/A')
        print(f"  rotation = {rotation_from_container}")
        
        # 方法 3: 從 side_data 取得 (較新的 PyAV 版本)
        print(f"\n[方法 3: stream.side_data]")
        rotation_from_side_data = None
        if hasattr(stream, 'side_data'):
            print(f"  side_data 屬性存在")
            try:
                side_data_list = list(stream.side_data)
                print(f"  side_data 數量: {len(side_data_list)}")
                for i, side_data in enumerate(side_data_list):
                    print(f"\n  [{i}] side_data:")
                    print(f"      type: {side_data.type}")
                    print(f"      type.name: {side_data.type.name}")
                    # 印出所有可用屬性
                    attrs = [attr for attr in dir(side_data) if not attr.startswith('_')]
                    print(f"      可用屬性: {attrs}")
                    if side_data.type.name == 'DISPLAYMATRIX':
                        # 方法 3a: 直接使用 rotation 屬性（如果有）
                        if hasattr(side_data, 'rotation'):
                            rotation_from_side_data = side_data.rotation
                            print(f"      ✓ rotation (直接屬性): {rotation_from_side_data}")
                        
                        # 方法 3b: 使用 np.arctan2 從 display matrix 計算
                        if hasattr(side_data, 'to_ndarray'):
                            try:
                                matrix = np.array(side_data.to_ndarray())
                                print(f"      display matrix shape: {matrix.shape}")
                                print(f"      display matrix:\n{matrix}")
                                # 從矩陣計算旋轉角度
                                rotation_calculated = int(round(
                                    np.degrees(np.arctan2(matrix[1][0], matrix[0][0]))
                                ))
                                print(f"      ✓ rotation (arctan2 計算): {rotation_calculated}°")
                                rotation_from_side_data = rotation_calculated
                            except Exception as e:
                                print(f"      ⚠ 從 display matrix 計算 rotation 時錯誤: {e}")
                        else:
                            print(f"      ⚠ DISPLAYMATRIX 沒有 to_ndarray 方法")
            except Exception as e:
                print(f"  讀取 side_data 時錯誤: {e}")
        else:
            print(f"  side_data 屬性不存在 (此版本不支援)")
        
        # 方法 4: 嘗試從 codec context 取得
        print(f"\n[方法 4: codec_context]")
        codec_ctx = stream.codec_context
        if hasattr(codec_ctx, 'side_data'):
            print(f"  codec_context.side_data 存在")
            try:
                for side_data in codec_ctx.side_data:
                    print(f"    type: {side_data.type.name}")
            except Exception as e:
                print(f"    讀取時錯誤: {e}")
        else:
            print(f"  codec_context.side_data 不存在")
        
        # 測試讀取一幀並檢查實際尺寸
        print(f"\n[測試解碼]")
        for frame in container.decode(video=0):
            print(f"  解碼後幀尺寸: {frame.width} x {frame.height}")
            # 將幀轉換為 numpy array
            img = frame.to_ndarray(format='bgr24')
            print(f"  Numpy array shape: {img.shape} (height, width, channels)")
            break
        
        container.close()
        
        # 方法 5: 使用 ffprobe 指令讀取 (最可靠的方式)
        print(f"\n[方法 5: ffprobe 指令]")
        rotation_from_ffprobe, ffprobe_info = get_rotation_via_ffprobe(video_path)
        if ffprobe_info:
            print(f"  寬度: {ffprobe_info.get('width')}")
            print(f"  高度: {ffprobe_info.get('height')}")
            print(f"  tags: {ffprobe_info.get('tags')}")
        if rotation_from_ffprobe != 0:
            print(f"  ✓ rotation (ffprobe): {rotation_from_ffprobe}°")
        else:
            print(f"  rotation: 0° (無旋轉或無法偵測)")
        
        # 總結
        print(f"\n[總結]")
        if rotation_from_ffprobe != 0:
            print(f"  ✓ 從 ffprobe 成功取得 rotation: {rotation_from_ffprobe}°")
        elif rotation_from_stream != 'N/A':
            print(f"  ✓ 從 stream.metadata 成功取得 rotation: {rotation_from_stream}")
        elif rotation_from_container != 'N/A':
            print(f"  ✓ 從 container.metadata 成功取得 rotation: {rotation_from_container}")
        elif rotation_from_side_data is not None:
            print(f"  ✓ 從 frame.side_data DISPLAYMATRIX 成功取得 rotation: {rotation_from_side_data}°")
        else:
            print(f"  ⚠ 無法從任何來源取得 rotation，可能無旋轉")
            print(f"    （現代 iPhone 影片常使用 display matrix 而非 rotate metadata）")
        
        print(f"\n{'='*60}\n")
        
    except Exception as e:
        print(f"錯誤: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python test_rotation.py <video_path>")
        print("範例: python test_rotation.py D:\\videos\\test.MOV")
        sys.exit(1)
    
    test_rotation(sys.argv[1])
