from ultralytics import YOLO

IMAGE_SIZE = 1216

def train_model():
    model = YOLO("best.pt")
    results = model.train(data = "data.yaml", epochs = 100, imgsz = IMAGE_SIZE, batch = 2)

def predict_model():
    model = YOLO("runs/detect/train8/weights/best.pt")
    model.predict('datasets/test/images/ArduinoMega_Top_jpg.rf.4296818296b50ae297ac4efaa75b8df9.jpg', save=True, imgsz = IMAGE_SIZE, conf = 0.5)


if __name__ == "__main__":
    train_model()