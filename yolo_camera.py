import cv2
import numpy as np
import time
import torch
 
# 加载模型
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model = torch.hub.load('yolov5', 'custom', 'yolov5s.pt', source='local')
# 设置摄像头
cap = cv2.VideoCapture(0)
 
# 检测循环
while True:
    # 从摄像头读取图像
    ret, frame = cap.read()
    
    # 将图像转换为RGB格式
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 检测目标
    results = model(rgb_frame)
    
    # 将结果绘制在图像上
    results.render()
    annotated_frame = results.ims[0]
    
    # 显示结果
    cv2.imshow('YOLO Object Detection', annotated_frame)
    
    # 退出循环
    if cv2.waitKey(1) == ord('q'):
        break
 
# 释放摄像头和窗口资源
cap.release()
cv2.destroyAllWindows()