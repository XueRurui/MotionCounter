# 运动类型配置
EXERCISE_TYPES = {
    1: {'name': '俯卧撑', 'class_name': 'push_down', 'enter_threshold': 6, 'exit_threshold': 4},
    2: {'name': '深蹲', 'class_name': 'squat_down', 'enter_threshold': 6, 'exit_threshold': 4}
}

def save_exercise_types(types):
    """保存运动类型到配置文件"""
    with open(__file__, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    start_idx = None
    end_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith('EXERCISE_TYPES = {'):
            start_idx = i
        if start_idx is not None and line.strip() == '}':
            end_idx = i
            break

    if start_idx is not None and end_idx is not None:
        new_content = 'EXERCISE_TYPES = {\n'
        for key, value in types.items():
            new_content += f"    {key}: {{'name': '{value['name']}', 'class_name': '{value['class_name']}', 'enter_threshold': {value['enter_threshold']}, 'exit_threshold': {value['exit_threshold']}}},\n"
        new_content += '}\n'

        lines[start_idx:end_idx+1] = [new_content]

        with open(__file__, 'w', encoding='utf-8') as f:
            f.writelines(lines)

# 路径配置
PATHS = {
    'input_images': 'DataSet_Input_Photo',
    'output_images': 'DataSet_Output_Photo',
    'output_csv': 'DataSet_Output_CSV',
    'output_video': 'Output_Video',
    'output_camera': 'Output_Camera'
}

# 视频配置
VIDEO_CONFIG = {
    'fps': 24,
    'width': 640,
    'height': 480,
    'codec': 'mp4v'
}

# 分类器配置
CLASSIFIER_CONFIG = {
    'top_n_by_max_distance': 30,
    'top_n_by_mean_distance': 10
}

# 平滑配置
SMOOTHING_CONFIG = {
    'window_size': 10,
    'alpha': 0.2
}
