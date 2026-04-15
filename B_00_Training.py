import A_00_Tools_PostureEncoding as pe
import A_01_Tools_Classification as pc
import csv
import os
import cv2
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import pose as mp_pose
from PIL import Image
from PIL import ImageDraw
from matplotlib import pyplot as plt
import tqdm
import numpy as np

def visualize_and_save_poses(input_images_folder, output_visualization_folder, path_to_output_visualization_csv):
    """
    对输入图像文件夹中的每个图像进行姿势检测，将检测结果可视化并保存在指定的输出文件夹中，
    同时生成 CSV 文件保存关键点信息，CSV 文件的大目录路径由参数指定。

    :param input_images_folder:输入图像文件夹的路径
    :param output_visualization_folder:输出可视化图像的文件夹路径。
    :param path_to_output_visualization_csv: 输出 CSV 文件的大目录路径。
    :return:
    """

    # 如果输出可视化结果的文件夹不存在，则创建
    if not os.path.exists(output_visualization_folder):
        os.makedirs(output_visualization_folder)

    # 遍历输入图像文件夹中的每个姿势类别
    for pose_class_name in os.listdir(input_images_folder):
        pose_class_path = os.path.join(input_images_folder, pose_class_name)
        output_class_visualization_folder = os.path.join(output_visualization_folder, pose_class_name)
        if not os.path.exists(output_class_visualization_folder):
            os.makedirs(output_class_visualization_folder)

        # 遍历每个姿势类别中的每张图像
        for image_name in os.listdir(pose_class_path):
            input_image_path = os.path.join(pose_class_path, image_name)
            output_image_path = os.path.join(output_class_visualization_folder, image_name)

            # 加载图像
            input_frame = cv2.imread(input_image_path)
            input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)

            # 处理图像并保存可视化结果
            with mp_pose.Pose() as pose_tracker:
                result = pose_tracker.process(image=input_frame)
                pose_landmarks = result.pose_landmarks

            output_frame = input_frame.copy()
            if pose_landmarks is not None:
                mp_drawing.draw_landmarks(
                    image=output_frame,
                    landmark_list=pose_landmarks,
                    connections=mp_pose.POSE_CONNECTIONS)

            output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_image_path, output_frame)

    # 生成CSV文件
    generate_csv_files(output_visualization_folder, path_to_output_visualization_csv)


def generate_csv_files(output_visualization_folder, path_to_output_visualization_csv):
    """
    生成CSV文件，记录每张图像的文件名和对应的关键点坐标

    :param output_visualization_folder:输出可视化结果的文件夹路径
    :param path_to_output_visualization_csv:输出CSV文件的路径
    :return: null
    """
    # 如果输出CSV文件的文件夹不存在，则创建
    if not os.path.exists(path_to_output_visualization_csv):
        os.makedirs(path_to_output_visualization_csv)

    # 遍历输出可视化结果的文件夹中的每个姿势类别
    for pose_class_name in os.listdir(output_visualization_folder):
        output_class_visualization_folder = os.path.join(output_visualization_folder, pose_class_name)
        output_csv_path = os.path.join(path_to_output_visualization_csv, pose_class_name + '.csv')

        # 打开CSV文件以写入数据
        with open(output_csv_path, 'w', newline='') as csv_out_file:
            csv_out_writer = csv.writer(csv_out_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)

            # 获取该姿势类别下的所有图像文件名，并按字母顺序排序
            image_names = sorted([n for n in os.listdir(output_class_visualization_folder) if not n.startswith('.')])

            # 遍历每张图像
            for image_name in tqdm.tqdm(image_names):
                # 加载图像
                output_frame = cv2.imread(os.path.join(output_class_visualization_folder, image_name))
                output_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)

                # 处理图像并保存关键点坐标
                with mp_pose.Pose() as pose_tracker:
                    result = pose_tracker.process(image=output_frame)
                    pose_landmarks = result.pose_landmarks

                if pose_landmarks is not None:
                    frame_height, frame_width = output_frame.shape[0], output_frame.shape[1]
                    pose_landmarks = np.array(
                        [[lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width]
                         for lmk in pose_landmarks.landmark],
                        dtype=np.float32)
                    assert pose_landmarks.shape == (33, 3), 'Unexpected landmarks shape: {}'.format(
                        pose_landmarks.shape)

                    # 将图像文件名和关键点坐标写入CSV文件
                    csv_out_writer.writerow([image_name] + pose_landmarks.flatten().astype(np.str_).tolist())

# # 指定输入图像文件夹路径、输出可视化结果的文件夹路径和输出CSV文件的路径
# input_images_folder = 'DataSet_Input_Photo'
# output_visualization_folder = 'DataSet_Output_Photo'
# path_to_output_visualization_csv = 'DataSet_Output_CSV'
#
# # 调用函数进行图像处理和可视化
# visualize_and_save_poses(input_images_folder, output_visualization_folder, path_to_output_visualization_csv)
