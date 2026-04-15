
import B_00_Training as tp
import C_02_Show_Camera as vc
import C_01_Show_Video as vp
from config import EXERCISE_TYPES, PATHS


def get_valid_input(prompt, valid_range):
    while True:
        try:
            value = int(input(prompt))
            if value in valid_range:
                return value
            print(f"请输入 {min(valid_range)} 到 {max(valid_range)} 之间的数字")
        except ValueError:
            print("输入无效，请输入数字")


def main():
    while True:
        print("\n=== 运动计数系统 ===")
        mode = get_valid_input("1. 视频检测\n2. 摄像头检测\n3. 训练数据\n4. 退出\n请选择功能: ", [1, 2, 3, 4])

        if mode == 4:
            print("退出程序")
            break

        if mode in [1, 2]:
            print(f"请确保 {PATHS['output_csv']} 目录中存在配置文件")
            exercise_options = "\n".join([f"{k}. {v['name']}" for k, v in EXERCISE_TYPES.items()])
            flag = get_valid_input(f"{exercise_options}\n请选择运动类型: ", list(EXERCISE_TYPES.keys()))

            try:
                if mode == 1:
                    video_path = input("请输入视频路径: ").strip()
                    if not video_path:
                        print("视频路径不能为空")
                        continue
                    vp.video_process(video_path, flag)
                else:
                    print("按 q 或 Esc 退出")
                    vc.process(flag)
            except Exception as e:
                print(f"处理失败: {e}")

        elif mode == 3:
            print(f"请将运动图片放入 {PATHS['input_images']} 目录")
            try:
                tp.visualize_and_save_poses(
                    PATHS['input_images'],
                    PATHS['output_images'],
                    PATHS['output_csv']
                )
                print(f"训练完成\n图片保存在: {PATHS['output_images']}\n配置保存在: {PATHS['output_csv']}")
            except Exception as e:
                print(f"训练失败: {e}")


if __name__ == '__main__':
    main()