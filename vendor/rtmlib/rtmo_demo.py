# import time

# import cv2
# import imageio

from rtmlib.tools.solution.body import Body
from rtmlib.visualization import draw_skeleton

# import numpy as np

def rtmo_body(video_path, output_path, start_frame=0, end_frame=-1):
    device = 'cuda'
    backend = 'onnxruntime'  # opencv, onnxruntime, openvino

    cap = cv2.VideoCapture(video_path)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    openpose_skeleton = False  # True for openpose-style, False for mmpose-style

    body = Body(
        pose='rtmo',
        to_openpose=openpose_skeleton,
        mode='balanced',  # balanced, performance, lightweight
        backend=backend,
        device=device)

    frame_idx = 0
    frames = []

    while cap.isOpened():
        success, frame = cap.read()
        frame_idx += 1

        if not success or (frame_idx > end_frame and end_frame > 0):
            break
        if frame_idx >= start_frame:
            s = time.time()
            keypoints, scores = body(frame)
            print(scores)
            det_time = time.time() - s
            print('det: ', det_time)

            img_show = frame.copy()

            # if you want to use black background instead of original image,
            # img_show = np.zeros(img_show.shape, dtype=np.uint8)

            img_show = draw_skeleton(img_show,
                                    keypoints,
                                    scores,
                                    openpose_skeleton=openpose_skeleton,
                                    kpt_thr=0.3,
                                    line_width=2)

            img_show = cv2.resize(img_show, (original_width, original_height))
            img_show_rgb = cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)
            frames.append(img_show_rgb)
            # cv2.imshow('img', img_show)
            # cv2.waitKey(10)
    imageio.mimsave(output_path, frames, fps=30)

if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), './rtmlib/rtmlib'))
    import time

    import cv2
    import imageio

    start_timestamp = time.time()
    video_path = "20240910_055820000_iOS.MOV"
    output_path = "output1.gif"
    start_frame = 2780  # Starting frame number
    end_frame = 3260  # Ending frame number
    rtmo_body(video_path, output_path, start_frame, end_frame)
    end_timestamp = time.time()
    print("程式執行時間：", end_timestamp-start_timestamp, "秒")


# if __name__ == "__main__":
#     import yaml 
#     import os
#     # Load configuration parameters from config.yaml
#     with open('config.yaml', 'r') as f:
#         config = yaml.safe_load(f)

#     data_config = config.get("data", {})
#     input_path = data_config.get("video_path", "")
#     output_base = data_config.get("output_path", "")

#     start_time = time.time()

#     if os.path.isdir(input_path):
#         for root, dirs, files in os.walk(input_path):
#             for filename in files:
#                 if filename.lower().endswith(('.mov', '.mp4', '.avi', '.mkv')):
#                     video_file = os.path.join(root, filename)
#                     rel_path = os.path.relpath(root, input_path)
#                     output_folder = os.path.join(output_base, rel_path)
#                     os.makedirs(output_folder, exist_ok=True)
#                     base_name = os.path.splitext(filename)[0]
#                     output_file = os.path.join(output_folder, base_name + ".gif")
#                     print(f"Processing: {video_file} -> {output_file}")
#                     rtmo_body(video_file, output_file)
#     elif os.path.isfile(input_path):
#         print(f"Processing single file: {input_path}")
#         rtmo_body(input_path, output_base, start_frame, end_frame)
#     else:
#         print(f"Input path does not exist: {input_path}")

#     end_time = time.time()
#     print("Execution time:", end_time - start_time, "seconds")
