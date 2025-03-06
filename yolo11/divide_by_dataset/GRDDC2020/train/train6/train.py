import warnings
import sys
import argparse
import os
sys.path.append('/root/ultralyticsPro/')
warnings.filterwarnings('ignore')  # 忽略警告信息，避免干扰
from ultralytics import YOLO

# 配置固定的常规超参数
lr0 = 1e-3  # 初始学习率
momentum = 0.9  # 动量
weight_decay = 5e-4  # 权重衰减
batch = 128  # 批次大小
epochs = 150  # 训练轮数
imgsz = 640  # 图片大小
device = 0  # 使用的设备，若有GPU，建议改为'cuda'
optimizer = 'AdamW'  # 优化器
workers = 8  # 数据加载的线程数
cache = True  # 是否缓存数据

# 配置YOLO模型
model = YOLO('/root/YOLOv11/yolo11n.pt')  # 加载模型配置

# 启动训练
results = model.train(
    data='/root/autodl-tmp/data.yaml',  # 数据集yaml文件路径
    imgsz=imgsz,  # 图片大小
    epochs=epochs,  # 训练的轮数
    batch=batch,  # 每批次的样本数
    workers=workers,  # 数据加载的线程数
    device=device,  # 使用的设备，若有GPU，建议改为'cuda'
    optimizer=optimizer,  # 优化器
    amp=True,  # 自动混合精度训练，若有GPU支持可以设为True
    cache=cache,  # 是否缓存数据
    lr0=lr0,  # 初始学习率
    lrf=0.1,  # 学习率衰减因子
    momentum=momentum,  # 动量
    weight_decay=weight_decay,  # 权重衰减 (L2正则化)
    warmup_epochs=3,  # 预热轮数
    warmup_momentum=0.8,  # 预热阶段动量
    warmup_bias_lr=0.1,  # 预热阶段偏置学习率
)

# 输出训练结果
print("Training completed!")
