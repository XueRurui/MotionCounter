# 运动计数系统 (Motion Counter)

基于 MediaPipe 的智能运动计数系统，支持俯卧撑、深蹲等运动的自动识别和计数。

## 功能特点

- 🎯 实时运动识别和计数
- 📹 支持视频文件和摄像头输入
- 🖥️ 图形界面和命令行双模式
- 🔧 可扩展的运动类型配置
- 📊 姿态可视化和数据导出

该系统结合了机器学习、图像处理和姿势检测技术，提供了一个实时的、可视化的运 动监测和分类方案。系统设计在于使用摄像头或导入本地视频，运用 MediaPipe 三维实时 人体关键点提供手势、人体姿势、人脸等识别和追踪功能，结合 KNN 最邻近分类算法、 指数移动平均方法等相关算法与方法，利用Python 语言进行内容编写。对各帧图片进行描 点、分类、平滑处理，得到相关数据后将各帧图片重新组合成视频。在视频左上角基于  PIL（Pillow）和 Matplotlib 的姿势分类结果，实现一个用于可视化姿势分类结果的工具。 通过绘制分类历史和计数器，直观展示姿势的置信度随时间的变化，并给予用户反馈动作 是否标准。

## 系统要求

- Python >= 3.9

## 安装

1. 克隆项目

```bash
git clone https://github.com/XueRurui/MotionCounter.git
cd MotionCounter
```

2. 安装依赖

```bash
conda create --name motioncounter python=3.9
```

```bash
pip install -r requirements.txt
```

## 使用方法

### 图形界面（推荐）

```
python gui.py
```

### 命令行界面

```bash
python main.py
```

## 快速开始

### 1. 训练数据

将运动姿态图片放入 `DataSet_Input_Photo` 目录，按以下格式命名：

- 俯卧撑: `push_down_001.jpg`, `push_up_001.jpg`
- 深蹲: `squat_down_001.jpg`, `squat_up_001.jpg`

运行训练生成配置文件。

### 2. 检测运动

选择视频文件或启动摄像头，系统将自动识别并计数。

## 项目结构

```txt
motion-counter/
├── gui.py                          # 图形界面
├── main.py                         # 命令行界面
├── config.py                       # 配置文件
├── video_processor.py              # 视频处理核心
├── A_00_Tools_PostureEncoding.py   # 姿态编码
├── A_01_Tools_Classification.py    # 姿态分类
├── A_02_Tools_NoiseReduction.py    # 结果平滑
├── A_03_Tools_Counter.py           # 动作计数
├── B_00_Training.py                # 训练模块
├── C_00_Show_Window.py             # 可视化
├── C_01_Show_Video.py              # 视频处理
├── C_02_Show_Camera.py             # 摄像头处理
├── DataSet_Input_Photo/            # 训练图片输入
├── DataSet_Output_Photo/           # 训练图片输出
├── DataSet_Output_CSV/             # 配置文件
├── Output_Video/                   # 视频输出
└── Output_Camera/                  # 摄像头输出
```

## 添加新运动类型

### 方法一：图形界面（推荐）

1. 启动图形界面 `python gui.py`
2. 点击"管理运动类型"按钮
3. 点击"添加"按钮
4. 填写运动信息：
   - ID：运动类型编号（自动生成）
   - 名称：显示名称（如"仰卧起坐"）
   - 类名：分类器使用的类名（如"situp_down"）
   - 进入阈值：判定进入动作的阈值（默认6）
   - 退出阈值：判定退出动作的阈值（默认4）
5. 点击"保存"完成添加

### 方法二：手动编辑配置文件

在 `config.py` 中添加配置：

```python
EXERCISE_TYPES = {
    1: {'name': '俯卧撑', 'class_name': 'push_down', 'enter_threshold': 6, 'exit_threshold': 4},
    2: {'name': '深蹲', 'class_name': 'squat_down', 'enter_threshold': 6, 'exit_threshold': 4},
    3: {'name': '仰卧起坐', 'class_name': 'situp_down', 'enter_threshold': 6, 'exit_threshold': 4}
}
```

## 技术原理

1. 姿态检测: 使用 MediaPipe Pose 提取 33 个关键点。MediaPipe Pose 是 Google 开发的一款用于实时人体姿态估计的算法。简单来说，它能通过普通摄像头识别出人体33个关键骨骼点，连接起来构成骨架，从而理解人的动作和姿态。
2. 姿态编码: 计算关键点间的距离向量作为特征。代码中定义了一个名为 FullBodyPoseEmbedder 的类，用于将人体姿态的3D关键点（landmarks）转换为一个标准化的嵌入表示。它首先通过构造函数初始化一个躯干尺寸乘数参数，并预定义了人体33个关键点的名称列表。当输入一组关键点坐标后，该类会执行标准化操作：先计算左右髋部的中点作为姿势中心，将关键点平移到该中心；再根据躯干长度（肩部中点与髋部中点的距离）乘上乘数，以及关键点到姿势中心的最大距离，取两者中的较大值作为姿势的尺寸，并以此缩放所有关键点坐标。完成标准化后，该类会计算一系列成对关键点之间的三维向量差（即带符号的距离），这些距离覆盖了身体各主要关节的连接（如肩到肘、肘到腕、髋到膝等）、多关节组合（如肩到腕、髋到踝）、对称关节之间的交叉距离（如左右肘、左右腕），最终将这些向量拼接成一个嵌入数组输出。
3. KNN 分类: 基于距离匹配最相似的姿态。KNN 最邻近分类算法的实现原理：为了判断未知样本的类别，以所有已知类别的样本 作为参照，计算未知样本与所有已知样本的距离，从中选取与未知样本距离最近的 K 个已 知样本，根据少数服从多数的投票法则（majority-voting），将未知样本与 K 个最邻近样本中所属类别占比较多的归为一类。
4. EMA 平滑: 指数移动平均消除抖动，指数移动平均算法（EMA）是一种常用的时间序列数据平滑技术，用于消除数据中的噪 音和波动，从而突出趋势。与简单移动平均（SMA）不同，EMA 更加重视最近的数据点， 因此对于新数据的响应更快
5. 阈值计数: 双阈值机制避免误计数

## 致谢

基于 Google MediaPipe 姿态识别技术开发。

