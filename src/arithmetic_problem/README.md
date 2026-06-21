# arithmetic_problem ROS 2 接口与操作说明

`arithmetic_problem` 是一个 ROS 2 C++ 节点包，用于从相机采集图像，识别图像中的数学表达式，完成计算后发布一个整数结果。

## 节点信息

| 项目 | 内容 |
| --- | --- |
| 包名 | `arithmetic_problem` |
| 可执行文件 | `arithmetic_problem` |
| 默认节点名 | `arithmetic_problem_node` |
| 启动方式 | `ros2 run arithmetic_problem arithmetic_problem` |

当前包没有 launch 文件，主要通过 `ros2 run` 启动，并通过 ROS 2 参数触发一次识别计算任务。

## 发布话题

| 话题 | 类型 | QoS | 说明 |
| --- | --- | --- | --- |
| `/vip_box_id` | `std_msgs/msg/Int32` | depth 1, reliable, transient local | 每次识别计算任务完成后发布计算结果 |

`vip_box_id` 是相对话题名。如果节点运行在命名空间下，实际话题会带上对应命名空间。

查看结果：

```bash
ros2 topic echo /vip_box_id
```

查看话题 QoS 和连接信息：

```bash
ros2 topic info -v /vip_box_id
```

## 参数接口

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `camera_config_path` | `string` | `camera_driver/camera_init/HIKcamera0.yaml` | 相机初始化 YAML 路径。可由环境变量 `ARITHMETIC_CAMERA_CONFIG` 覆盖默认值 |
| `onnx_model_path` | `string` | `src/best.onnx` | 数学字符检测 ONNX 模型路径。可由环境变量 `ARITHMETIC_ONNX_MODEL` 覆盖默认值 |
| `model_input_width` | `int` | `640` | ONNX 模型输入宽度 |
| `model_input_height` | `int` | `640` | ONNX 模型输入高度 |
| `run_with_cuda` | `bool` | `true` | 是否使用 OpenCV DNN CUDA 后端 |
| `show_window` | `bool` | `false` | 是否显示识别调试窗口 |
| `min_samples` | `int` | `20` | 至少采集多少个有效样本后才允许输出稳定结果 |
| `max_samples` | `int` | `100` | 单次任务最多采集多少个有效样本 |
| `dominance_threshold` | `double` | `0.80` | 众数占比超过该阈值时提前结束采样 |
| `timeout_ms` | `int` | `5000` | 单次任务最长采样时间，单位 ms |
| `start_calc` | `bool` | `false` | 触发参数。设为 `true` 时启动一次识别计算任务 |

`start_calc` 必须是 bool 类型。节点完成一次任务后，会尝试自动把它重置为 `false`。

## 启动方法

编译包：

```bash
colcon build --packages-select arithmetic_problem
source install/setup.bash
```

使用默认配置启动：

```bash
ros2 run arithmetic_problem arithmetic_problem
```

启动时指定常用参数：

```bash
ros2 run arithmetic_problem arithmetic_problem --ros-args \
  -p camera_config_path:=/home/lyz/桌面/arm/src/arithmetic_problem/camera_driver/camera_init/HIKcamera0.yaml \
  -p onnx_model_path:=/home/lyz/桌面/arm/src/arithmetic_problem/src/best.onnx \
  -p run_with_cuda:=true \
  -p timeout_ms:=5000
```

如果机器没有可用 CUDA，建议改为 CPU 推理：

```bash
ros2 run arithmetic_problem arithmetic_problem --ros-args -p run_with_cuda:=false
```

也可以通过环境变量指定默认路径：

```bash
export ARITHMETIC_CAMERA_CONFIG=/home/lyz/桌面/arm/src/arithmetic_problem/camera_driver/camera_init/HIKcamera0.yaml
export ARITHMETIC_ONNX_MODEL=/home/lyz/桌面/arm/src/arithmetic_problem/src/best.onnx
ros2 run arithmetic_problem arithmetic_problem
```

## 触发一次识别计算

节点启动后不会持续发布结果，需要外部设置 `start_calc=true` 触发一次任务：

```bash
ros2 param set /arithmetic_problem_node start_calc true
```

推荐在另一个终端同时监听结果：

```bash
source install/setup.bash
ros2 topic echo /vip_box_id
```

一次完整操作流程：

```bash
# 终端 1
source install/setup.bash
ros2 run arithmetic_problem arithmetic_problem

# 终端 2
source install/setup.bash
ros2 topic echo /vip_box_id

# 终端 3
source install/setup.bash
ros2 param set /arithmetic_problem_node start_calc true
```

如果上一轮任务仍在执行，新触发会被节点丢弃。等待 `start_calc` 回到 `false` 或日志显示结果已发布后，再触发下一次任务。

## 运行时修改参数

参数可以在节点运行时修改。新的配置会在下一次触发计算时生效：

```bash
ros2 param set /arithmetic_problem_node timeout_ms 8000
ros2 param set /arithmetic_problem_node min_samples 30
ros2 param set /arithmetic_problem_node dominance_threshold 0.75
ros2 param set /arithmetic_problem_node show_window true
```

查看当前参数：

```bash
ros2 param list /arithmetic_problem_node
ros2 param get /arithmetic_problem_node camera_config_path
ros2 param get /arithmetic_problem_node onnx_model_path
```

## 相机配置文件

`camera_config_path` 指向 OpenCV YAML 文件。常用字段如下：

```yaml
camera_type: 1       # USB 为 0，海康相机为 1
base_info:
  width: 864
  height: 800
  fps: 100
pixel_format: 35127316
hik_info:
  camera_mode: 1
  exposure_time: 10000
```

默认配置位于：

```text
src/arithmetic_problem/camera_driver/camera_init/HIKcamera0.yaml
src/arithmetic_problem/camera_driver/camera_init/USBcamera0.yaml
src/arithmetic_problem/camera_driver/camera_init/USBcamera1.yaml
```

## 注意事项

- 本节点没有订阅话题、普通服务或 action；对外控制入口主要是 ROS 2 参数服务。
- `vip_box_id` 使用 transient local QoS，后启动的订阅者通常可以收到最近一次发布的结果。
- `show_window=true` 需要图形界面环境；无显示环境运行时保持 `false`。
- 默认路径来自编译时源码目录。部署到其他机器或目录后，建议显式传入 `camera_config_path` 和 `onnx_model_path`。
- 如果没有结果输出，优先检查相机是否打开、模型路径是否正确、`run_with_cuda` 是否适合当前机器，以及是否已经设置 `start_calc=true`。
