from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/usr/local/deeplearn/yangqu/model/yolov8s.pt')
    model.train(
        data='/usr/local/deeplearn/test/ygytest/model/clo-man-hat3.yaml',
        epochs=150,
        batch=64,
        #cos_lr=True,
        #lr0=0.001,           # SGD 默
        #lrf=0.01,
        #weight_decay=0.001,
        #mosaic=0.5,
        #close_mosaic=10,
        #mixup=0,
        #scale=0.5,
        #degrees=10,
        #translate=0.1,
        #dropout=0.05,
        device=0,
    )

