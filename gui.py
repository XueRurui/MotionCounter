import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import B_00_Training as tp
import C_02_Show_Camera as vc
import C_01_Show_Video as vp
from config import EXERCISE_TYPES, PATHS, save_exercise_types
import importlib
import config


class MotionCounterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("运动计数系统")
        self.root.geometry("500x450")
        self.root.resizable(False, False)
        self.exercise_types = EXERCISE_TYPES.copy()

        self.create_widgets()

    def create_widgets(self):
        # 标题
        title = tk.Label(self.root, text="运动计数系统", font=("Arial", 20, "bold"))
        title.pack(pady=20)

        # 功能选择
        frame = ttk.LabelFrame(self.root, text="选择功能", padding=20)
        frame.pack(padx=20, pady=10, fill="both", expand=True)

        # 运动类型选择
        tk.Label(frame, text="运动类型:", font=("Arial", 11)).grid(row=0, column=0, sticky="w", pady=10)
        self.exercise_var = tk.IntVar(value=1)
        self.exercise_frame = tk.Frame(frame)
        self.exercise_frame.grid(row=0, column=1, sticky="w", pady=10)
        self.refresh_exercise_types()

        # 管理按钮
        ttk.Button(frame, text="管理运动类型", command=self.manage_exercises, width=15).grid(row=0, column=2, padx=10)

        # 按钮区域
        btn_frame = tk.Frame(frame)
        btn_frame.grid(row=1, column=0, columnspan=3, pady=20)

        ttk.Button(btn_frame, text="视频检测", command=self.video_detect, width=15).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="摄像头检测", command=self.camera_detect, width=15).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="训练数据", command=self.train_data, width=15).pack(side="left", padx=5)

        # 状态栏
        self.status_var = tk.StringVar(value="就绪")
        status_bar = tk.Label(self.root, textvariable=self.status_var,
                             relief=tk.SUNKEN, anchor="w")
        status_bar.pack(side="bottom", fill="x")

    def refresh_exercise_types(self):
        for widget in self.exercise_frame.winfo_children():
            widget.destroy()
        for key, value in self.exercise_types.items():
            ttk.Radiobutton(self.exercise_frame, text=value['name'],
                           variable=self.exercise_var, value=key).pack(side="left", padx=10)

    def manage_exercises(self):
        ManageExerciseWindow(self.root, self.exercise_types, self.on_exercise_updated)

    def on_exercise_updated(self, updated_types):
        self.exercise_types = updated_types
        self.refresh_exercise_types()
        if self.exercise_types:
            self.exercise_var.set(min(self.exercise_types.keys()))

    def video_detect(self):
        video_path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[("视频文件", "*.mp4 *.avi *.mov"), ("所有文件", "*.*")]
        )
        if not video_path:
            return

        self.status_var.set("处理中...")
        threading.Thread(target=self._process_video, args=(video_path,), daemon=True).start()

    def _process_video(self, video_path):
        try:
            vp.video_process(video_path, self.exercise_var.get())
            self.root.after(0, lambda: self.status_var.set("视频处理完成"))
            self.root.after(0, lambda: messagebox.showinfo("成功", "视频处理完成"))
        except Exception as e:
            self.root.after(0, lambda: self.status_var.set("处理失败"))
            self.root.after(0, lambda err=e: messagebox.showerror("错误", f"处理失败: {err}"))

    def camera_detect(self):
        self.status_var.set("摄像头运行中...")
        threading.Thread(target=self._process_camera, daemon=True).start()

    def _process_camera(self):
        try:
            vc.process(self.exercise_var.get())
            self.root.after(0, lambda: self.status_var.set("摄像头检测完成"))
        except Exception as e:
            self.root.after(0, lambda: self.status_var.set("处理失败"))
            self.root.after(0, lambda err=e: messagebox.showerror("错误", f"处理失败: {err}"))

    def train_data(self):
        result = messagebox.askyesno("确认", f"请确保训练图片已放入 {PATHS['input_images']} 目录\n是否继续?")
        if not result:
            return

        self.status_var.set("训练中...")
        threading.Thread(target=self._train, daemon=True).start()

    def _train(self):
        try:
            tp.visualize_and_save_poses(PATHS['input_images'], PATHS['output_images'], PATHS['output_csv'])
            self.root.after(0, lambda: self.status_var.set("训练完成"))
            self.root.after(0, lambda: messagebox.showinfo("成功",
                f"训练完成\n图片: {PATHS['output_images']}\n配置: {PATHS['output_csv']}"))
        except Exception as e:
            self.root.after(0, lambda: self.status_var.set("训练失败"))
            self.root.after(0, lambda err=e: messagebox.showerror("错误", f"训练失败: {err}"))


class ManageExerciseWindow:
    def __init__(self, parent, exercise_types, callback):
        self.window = tk.Toplevel(parent)
        self.window.title("管理运动类型")
        self.window.geometry("600x400")
        self.exercise_types = exercise_types.copy()
        self.callback = callback

        self.create_widgets()

    def create_widgets(self):
        # 列表框
        frame = tk.Frame(self.window)
        frame.pack(padx=10, pady=10, fill="both", expand=True)

        scrollbar = tk.Scrollbar(frame)
        scrollbar.pack(side="right", fill="y")

        self.listbox = tk.Listbox(frame, yscrollcommand=scrollbar.set, font=("Arial", 10))
        self.listbox.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.listbox.yview)

        self.refresh_list()

        # 按钮
        btn_frame = tk.Frame(self.window)
        btn_frame.pack(pady=10)

        ttk.Button(btn_frame, text="添加", command=self.add_exercise, width=12).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="编辑", command=self.edit_exercise, width=12).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="删除", command=self.delete_exercise, width=12).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="保存", command=self.save_changes, width=12).pack(side="left", padx=5)

    def refresh_list(self):
        self.listbox.delete(0, tk.END)
        for key, value in sorted(self.exercise_types.items()):
            self.listbox.insert(tk.END, f"{key}. {value['name']} ({value['class_name']})")

    def add_exercise(self):
        ExerciseEditWindow(self.window, None, self.on_exercise_saved)

    def edit_exercise(self):
        selection = self.listbox.curselection()
        if not selection:
            messagebox.showwarning("警告", "请选择要编辑的运动类型")
            return
        idx = selection[0]
        key = sorted(self.exercise_types.keys())[idx]
        ExerciseEditWindow(self.window, (key, self.exercise_types[key]), self.on_exercise_saved)

    def delete_exercise(self):
        selection = self.listbox.curselection()
        if not selection:
            messagebox.showwarning("警告", "请选择要删除的运动类型")
            return
        idx = selection[0]
        key = sorted(self.exercise_types.keys())[idx]
        if messagebox.askyesno("确认", f"确定删除 {self.exercise_types[key]['name']} 吗？"):
            del self.exercise_types[key]
            self.refresh_list()

    def on_exercise_saved(self, key, data):
        self.exercise_types[key] = data
        self.refresh_list()

    def save_changes(self):
        try:
            save_exercise_types(self.exercise_types)
            importlib.reload(config)
            self.callback(self.exercise_types)
            messagebox.showinfo("成功", "运动类型已保存")
            self.window.destroy()
        except Exception as e:
            messagebox.showerror("错误", f"保存失败: {e}")


class ExerciseEditWindow:
    def __init__(self, parent, exercise_data, callback):
        self.window = tk.Toplevel(parent)
        self.window.title("编辑运动类型" if exercise_data else "添加运动类型")
        self.window.geometry("400x300")
        self.callback = callback
        self.is_new = exercise_data is None

        if exercise_data:
            self.key, self.data = exercise_data
        else:
            self.key = None
            self.data = {'name': '', 'class_name': '', 'enter_threshold': 6, 'exit_threshold': 4}

        self.create_widgets()

    def create_widgets(self):
        frame = ttk.LabelFrame(self.window, text="运动信息", padding=20)
        frame.pack(padx=20, pady=20, fill="both", expand=True)

        tk.Label(frame, text="ID:").grid(row=0, column=0, sticky="w", pady=5)
        self.id_var = tk.IntVar(value=self.key if self.key else max(EXERCISE_TYPES.keys(), default=0) + 1)
        tk.Entry(frame, textvariable=self.id_var, state='disabled' if not self.is_new else 'normal').grid(row=0, column=1, sticky="ew", pady=5)

        tk.Label(frame, text="名称:").grid(row=1, column=0, sticky="w", pady=5)
        self.name_var = tk.StringVar(value=self.data['name'])
        tk.Entry(frame, textvariable=self.name_var).grid(row=1, column=1, sticky="ew", pady=5)

        tk.Label(frame, text="类名:").grid(row=2, column=0, sticky="w", pady=5)
        self.class_var = tk.StringVar(value=self.data['class_name'])
        tk.Entry(frame, textvariable=self.class_var).grid(row=2, column=1, sticky="ew", pady=5)

        tk.Label(frame, text="进入阈值:").grid(row=3, column=0, sticky="w", pady=5)
        self.enter_var = tk.IntVar(value=self.data['enter_threshold'])
        tk.Entry(frame, textvariable=self.enter_var).grid(row=3, column=1, sticky="ew", pady=5)

        tk.Label(frame, text="退出阈值:").grid(row=4, column=0, sticky="w", pady=5)
        self.exit_var = tk.IntVar(value=self.data['exit_threshold'])
        tk.Entry(frame, textvariable=self.exit_var).grid(row=4, column=1, sticky="ew", pady=5)

        frame.columnconfigure(1, weight=1)

        btn_frame = tk.Frame(self.window)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="保存", command=self.save, width=12).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="取消", command=self.window.destroy, width=12).pack(side="left", padx=5)

    def save(self):
        try:
            key = self.id_var.get()
            data = {
                'name': self.name_var.get().strip(),
                'class_name': self.class_var.get().strip(),
                'enter_threshold': self.enter_var.get(),
                'exit_threshold': self.exit_var.get()
            }
            if not data['name'] or not data['class_name']:
                messagebox.showwarning("警告", "名称和类名不能为空")
                return
            self.callback(key, data)
            self.window.destroy()
        except Exception as e:
            messagebox.showerror("错误", f"保存失败: {e}")


def main():
    root = tk.Tk()
    app = MotionCounterGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
