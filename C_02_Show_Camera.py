import datetime
import os
import cv2
import numpy as np
import C_00_Show_Window as vs
from video_processor import VideoProcessor
from config import EXERCISE_TYPES, PATHS, VIDEO_CONFIG


def process(flag):
    if flag not in EXERCISE_TYPES:
        raise ValueError(f"无效的运动类型: {flag}")

    exercise_config = EXERCISE_TYPES[flag]
    mkfile_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y年%m月%d日%H时%M分%S秒')

    os.makedirs(PATHS['output_camera'], exist_ok=True)
    out_video_path = f"./{PATHS['output_camera']}/{exercise_config['name']}{mkfile_time}.mp4"

    cv2.namedWindow('video', cv2.WINDOW_NORMAL)
    video_cap = cv2.VideoCapture(0)

    if not video_cap.isOpened():
        raise IOError("无法打开摄像头")

    try:
        processor = VideoProcessor(exercise_config)
        processor.visualizer = vs.PoseClassificationVisualizer(
            class_name=exercise_config['class_name'],
            plot_y_max=10
        )

        out_video = cv2.VideoWriter(
            out_video_path,
            cv2.VideoWriter_fourcc(*VIDEO_CONFIG['codec']),
            VIDEO_CONFIG['fps'],
            (VIDEO_CONFIG['width'], VIDEO_CONFIG['height'])
        )

        while video_cap.isOpened():
            success, input_frame = video_cap.read()
            if not success:
                break

            output_frame = processor.process_frame(input_frame)

            cv2.imshow('video', cv2.cvtColor(np.array(output_frame), cv2.COLOR_RGB2BGR))
            out_video.write(cv2.cvtColor(np.array(output_frame), cv2.COLOR_RGB2BGR))

            if cv2.waitKey(1) in [ord('q'), 27]:
                break

        print(f"视频处理完成，输出保存在: {out_video_path}")

    finally:
        out_video.release()
        video_cap.release()
        cv2.destroyAllWindows()
        processor.close()