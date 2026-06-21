# 这里列出了我的需求

- 创建一个main.cpp文件，首先请将主程序入口迁移至该文件，原有的文件就移动到include作为hpp文件引用，同时请删除冗余代码

- 请你创建一个calculate.cpp和calculate.h文件封装成库，我希望这个库参照arithmetic.cpp的流程实现原有功能。

  > 原有功能实现流程：
  >  1. 从工业相机获取图像
  >  2. 调用opencv dnn加载onnx模型（onnx推理相关库在“/home/yuan/Vscode_word/awork_mycode/arithmetic_problem/include/camera_driver/include/inference.h”），对图片进行推理得到推理结果outputs
  >  3. 从推理结果outpus中遍历每个得到矩形框output.box
  >  4. 检测结果为运算符号“+”、“-”、“×”、“÷”、“=”的框special_box，识别到之后筛选掉y坐标相差大于一个special_box.y的结果，筛选掉y坐标相似但该output.box.y（框的高）大于2*special_box.y的框
  >  5. 将所得结果按x坐标拼成算式计算，所得结果对4取模
  >  6. 用cout函数输出得到结果

- 请对以往程序运行计算算式的结果进行统计，最多纳入100项结果，最少20项，当计算结果中频率最高的一项占比大于30%或者程序运行超过5秒时，将计算结果中频率最高的一项返回，并使用cout在终端输出它在已经统计的结果中的占比

- 然后在main.cpp内部加入一个ros2的话题通信，话题为“vip_box_id”，希望你调用calculate库得到返回的计算结果，然后用话题通信将它发布

- 完善cmake文档