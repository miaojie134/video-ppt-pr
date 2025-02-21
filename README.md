# 视频 PPT 幻灯片变化检测工具

这是一个用于检测视频中 PPT 幻灯片变化的工具，采用多维特征分析和自适应阈值机制，实现高精度的幻灯片切换检测。

## 功能特点

- 智能检测机制：

  - 自适应阈值调整
  - 多维特征融合分析
  - 运动特征智能判断
  - 讲师动作干扰过滤
  - 文本结构变化分析

- 支持多种变化类型：

  - 突变检测（sudden）
  - 渐变检测（gradual）
  - 静态变化检测（static）
  - 运动变化检测（motion）
  - 淡入淡出检测（fade）

- 导出支持：
  - Premiere Pro CSV 标记文件
  - Final Cut Pro XML 文件

## 系统流程

1. **视频输入处理**：

   - 读取视频文件
   - 提取视频基本信息（帧率、总帧数等）
   - 初始化检测参数和状态

2. **帧级特征分析**：

   - 提取每帧的灰度图像
   - 计算直方图特征
   - 分析文本密度分布
   - 提取边缘特征
   - 计算亮度和对比度

3. **运动特征分析**：

   - 计算相邻帧光流特征
   - 分析运动幅度和方向
   - 检测讲师动作区域
   - 评估运动连贯性

4. **变化检测与判断**：

   - 计算特征变化分数
   - 应用自适应阈值
   - 分析特征历史记录
   - 确定变化类型
   - 过滤无效变化

5. **结果输出**：
   - 生成时间戳列表
   - 导出标记文件
   - 提供调试信息

## 安装

1. 克隆仓库：

```bash
git https://github.com/miaojie134/video-to-pr-xml.git
cd video-ppt
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

## 使用方法

基本用法：

```bash
python main.py video_path [选项]
```

参数说明：

- `video_path`：输入视频文件路径（必需）
- `-t, --threshold`：检测阈值，范围 0-1.0，默认 0.25
- `-i, --min-interval`：最小检测间隔（秒），默认 1.0
- `-o, --output`：输出文件路径（可选）
- `-d, --debug-timestamps`：需要调试输出的时间点或时间区间列表（可选）
  - 单个时间点格式：`HH:MM:SS:FF`
  - 时间区间格式：`HH:MM:SS:FF-HH:MM:SS:FF`
  - 可以同时指定多个时间点和区间

示例：

```bash
# 基本使用
python main.py lecture.mp4

# 调试单个时间点
python main.py lecture.mp4 -d "00:00:18:09"

# 调试时间区间（分析连续帧）
python main.py lecture.mp4 -d "00:20:25:18-00:20:25:20"

# 同时调试多个时间点和区间
python main.py lecture.mp4 -d "00:00:18:09" "00:20:25:18-00:20:25:20" "00:30:00:00"

# 使用多个参数
python main.py lecture.mp4 --threshold 0.25 --min-interval 1.0 -d "00:20:25:18-00:20:25:20"
```

## 检测原理

1. **特征提取**：

   - 直方图相关性分析
   - 文本密度和结构分析
   - 边缘特征提取
   - 光流特征分析
   - 运动特征分析

2. **智能判断机制**：

   - 自适应阈值调整
   - 特征分布分析
   - 历史记录追踪
   - 多维特征融合

3. **变化类型判断**：

   - 突变检测：基于相关性和密度变化
   - 渐变检测：基于运动特征和连贯性
   - 静态检测：基于文本结构和密度变化
   - 运动检测：基于大幅度运动特征
   - 淡入淡出：基于亮度和对比度变化

4. **优化策略**：
   - 讲师动作过滤
   - 运动幅度自适应
   - 时间间隔验证
   - 多阶段确认机制

## 调试输出信息

使用 `--debug-timestamps` 参数可以查看详细的特征信息：

- 基础特征：

  - 相关性
  - 密度变化
  - 边缘变化
  - 文本结构变化
  - 运动幅度

- 光流特征：

  - 是否为渐变
  - 运动连贯性
  - 是否为讲师动作
  - 讲师运动比例

- 文本结构：
  - 文本区域数量
  - 平均区域高度
  - 平均区域宽度
  - 总文本面积

## 项目结构

```
video-ppt/
├── src/
│   ├── analyzer/
│   │   ├── frame_analyzer.py     # 帧分析
│   │   └── optical_flow.py       # 光流分析
│   ├── detector/
│   │   └── slide_detector.py     # 幻灯片检测
│   ├── exporters/
│   │   ├── csv_exporter.py       # CSV导出
│   │   └── xml_exporter.py       # XML导出
│   └── utils/
│       └── time_utils.py         # 时间工具
├── main.py                       # 主程序
└── requirements.txt              # 项目依赖
```

## 注意事项

1. 视频要求：

   - 建议使用清晰的视频源
   - 避免剧烈的相机抖动
   - 保持稳定的录制环境

2. 检测优化：

   - 系统会自动适应不同视频特点
   - 支持多种变化类型的智能识别
   - 可通过调试模式分析具体帧的特征

3. 特殊情况处理：
   - 自动过滤讲师动作干扰
   - 智能处理大幅度运动
   - 支持静态内容变化检测

## 许可证

MIT License

## 打包说明

### Windows 打包步骤：

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 运行打包脚本：
```bash
create_exe.bat
```

3. 在 dist 文件夹中找到 video-ppt-detector.exe

### macOS 打包步骤：

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 添加执行权限：
```bash
chmod +x create_app.sh
```

3. 运行打包脚本：
```bash
./create_app.sh
```

4. 在 dist 文件夹中找到 video-ppt-detector.app
