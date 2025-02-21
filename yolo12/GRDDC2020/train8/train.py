import sys
import argparse
import os
from ultralytics import YOLO
def main(opt):
    yaml = opt.cfg
    weights = opt.weights
    # model = YOLO(weights)  # 直接加载权重文件进行训练
    # model = YOLO(yaml) # 加载自定义or默认的yaml配置文件
    model = YOLO(yaml).load(weights) # 加载yaml配置文件的同时，加载权重进行训练
    model.info()
    results = model.train(data='/root/autodl-tmp/data.yaml',  # 数据集yaml路径
                          epochs=150, 
                          batch=64, 
                          imgsz=640,
                          scale=0.9,  # S:0.9; M:0.9; L:0.9; X:0.9
                          mosaic=1.0,
                          mixup=0.15,  # S:0.05; M:0.15; L:0.15; X:0.2
                          copy_paste=0.4,  # S:0.15; M:0.4; L:0.5; X:0.6
                          device = 0,
                          cache = True,
                          optimizer = 'AdamW',
                          workers = 8
                        )

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='/root/autodl-tmp/yolov12/ultralytics/cfg/models/v12/yolov12.yaml', help='initial weights path')
    parser.add_argument('--weights', type=str, default='/root/autodl-tmp/yolov12/yolov12n.pt', help='')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)