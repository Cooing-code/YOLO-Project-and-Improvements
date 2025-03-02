import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'/root/autodl-tmp/ultralytics-main/ultralytics/cfg/models/11/yolo11-cls.yaml') 
    model.load('/root/autodl-tmp/ultralytics-main/yolo11n-cls.pt') 
    model.train(data=r"/root/autodl-tmp/data",#注意分类任务与检测任务数据集不同点
                task='classify',
                cache=True,
                imgsz=1024,
                epochs=100,
                batch=16,
                close_mosaic=0,
                workers=8,
                device=0,
                optimizer='AdamW',
                amp=True,
                project='runs/train',
                name='exp',
                )