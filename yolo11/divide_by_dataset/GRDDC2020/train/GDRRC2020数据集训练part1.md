#  GDRRC2020数据集训练

###  一.训练路径

1.原始数据集+超参数自适应优化-->2.增扩数据集（随机裁剪/旋转/对比度调整）-->3.增扩数据集+超参数自适应优化-->4.日本道路数据集

###  二.训练环境

预训练权重：yolov11n.pt/cocon_v27.pt

训练环境/平台：autodl/roboflow

###  三.训练结果

1.原始数据集+超参数自适应优化

![img](https://storage.googleapis.com/roboflow-platform-cache/EybDzjsUKBUnavj4x3Tw4xeBTDx2/RrWecipasXyUnh7pcnM1/2/results.png)

2.增扩数据集3x（随机裁剪/旋转/对比度调整）

![results](C:\Users\24965\Desktop\Xftp8DragDropSupportDir167620531\train10\results.png)

3.增扩数据集+超参数自适应优化

![img](https://storage.googleapis.com/roboflow-platform-cache/EybDzjsUKBUnavj4x3Tw4xeBTDx2/RrWecipasXyUnh7pcnM1/5/results.png)

4.分离独立日本道路数据集（对下文“未来优化方向（1）”的尝试）

![img](https://storage.googleapis.com/roboflow-platform-cache/EybDzjsUKBUnavj4x3Tw4xeBTDx2/RrWecipasXyUnh7pcnM1/1/results.png)

###  四.总结

1.最佳模型（增扩数据集+超参数自适应优化）：mAP：55,2%   置信度：57.7%   召回率：54.2%

2.未来优化方向：（1）观察可知，不同国家道路状况有较大差异，可尝试训练区基于各个国家道路损伤状况的独立模型，再在实际生产中训练区分各个国家道路特征的模型，提高模型的预测精确度（图片传入-->预测道路所属国家-->使用预测值对各个国家模型进行加权-->传入多国家独立模型预测-->对结果进行加权平均）

（2）使用更多超参数优化方法，提高超参数自适应调优次数与训练轮次（目前为自适应优化50轮+训练150轮）


