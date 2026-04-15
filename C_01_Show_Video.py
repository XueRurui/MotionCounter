import datetime
import os
import cv2
import numpy as np
import tqdm
import C_00_Show_Window as vs
from video_processor import VideoProcessor
from config import EXERCISE_TYPES, PATHS


def video_process(video_path, flag):
    if flag not in EXERCISE_TYPES:
        raise ValueError(f"无效的运动类型: {flag}")

    exercise_config = EXERCISE_TYPES[flag]
    mkfile_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y年%m月%d日%H时%M分%S秒')

    os.makedirs(PATHS['output_video'], exist_ok=True)
    out_video_path = f"./{PATHS['output_video']}/{exercise_config['name']}{mkfile_time}.mp4"

    video_cap = cv2.VideoCapture(video_path)
    if not video_cap.isOpened():
        raise IOError(f"无法打开视频: {video_path}")

    try:
        video_n_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = video_cap.get(cv2.CAP_PROP_FPS)
        video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        processor = VideoProcessor(exercise_config)
        processor.visualizer = vs.PoseClassificationVisualizer(
            class_name=exercise_config['class_name'],
            plot_x_max=video_n_frames,
            plot_y_max=10
        )

        out_video = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                     video_fps, (video_width, video_height))

        with tqdm.tqdm(total=video_n_frames, position=0, leave=True) as pbar:
            while True:
                success, input_frame = video_cap.read()
                if not success:
                    break

                output_frame = processor.process_frame(input_frame)
                out_video.write(cv2.cvtColor(np.array(output_frame), cv2.COLOR_RGB2BGR))
                pbar.update()

        print(f"视频处理完成，输出保存在: {out_video_path}")

    finally:
        out_video.release()
        video_cap.release()
        processor.close()