from ultralytics import YOLO
import cv2


# 加载 YOLO v10 预训练模型
try:
    model = YOLO("yolov10n.pt")
except Exception as e:
    print(f"Failed to load YOLO v10 model: {e}")
    exit(1)


# 设置摄像头
cap = cv2.VideoCapture(0)


# 检测循环
while True:
    # 从摄像头读取图像
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame from camera.")
        break
    
    # 检测目标
    results = model(frame)
    
    # 显示结果
    annotated_frame = results[0].plot()
    
    # 显示结果
    cv2.imshow('YOLO v10 Object Detection', annotated_frame)
    
    # 退出循环
    if cv2.waitKey(1) == ord('q'):
        break


# 释放摄像头和窗口资源
cap.release()
cv2.destroyAllWindows()