from ultralytics import YOLO
import cv2


IMAGE_SIZE = 1216

def train_model():
    model = YOLO("yolov8n.pt")
    results = model.train(data = "data.yaml", epochs = 600, imgsz = IMAGE_SIZE, batch = 2)

def predict_model():
    model = YOLO("best.pt")
    results = model.predict('datasets/evaluation', save=False, imgsz = IMAGE_SIZE, conf = 0.5)

    for pred in results:
        draw_prediction(pred.cpu())

def draw_prediction(prediction):
    image = cv2.imread(prediction.path)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    font_thickness = 2
    color = (0, 255, 0) 

    for bbox, label, confidence in zip(prediction.boxes.xyxy.numpy(), prediction.boxes.cls.numpy(), prediction.boxes.conf.numpy()):
        bbox = [ int(x) for x in bbox ]
        left, top, right, bottom = bbox
        cv2.rectangle(image, (left, top), (right, bottom), color, font_thickness)
        text = f"{prediction.names[label]}: {confidence:.2f}"
        cv2.putText(image, text, (left, top - 10), font, font_scale, color, font_thickness)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imshow('Predictions', image)

if __name__ == "__main__":
    predict_model()