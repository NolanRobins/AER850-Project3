from ultralytics import YOLO
import cv2

OUTPUT_DIR = "outputs/predictions/"
IMAGE_SIZE = 1216

def train_model():
    model = YOLO("yolov8m.pt")
    results = model.train(data = "data.yaml", epochs = 600, imgsz = IMAGE_SIZE, batch = 2)

def resume_training():
    model = YOLO("runs/detect/train - 600 epochs - yolo8m/weights/last.pt")
    results = model.train(data = "data.yaml", epochs = 600, imgsz = IMAGE_SIZE, batch = 2, resume = True)

def predict_model():
    model = YOLO("runs/detect/train - 600 epochs - yolo8m/weights/best.pt")
    results = model.predict('datasets/evaluation', save=True, imgsz = IMAGE_SIZE, conf = 0.1, max_det = 2000)

    # for pred in results:
    #     draw_prediction(pred.cpu())

def draw_prediction(prediction):
    image = cv2.imread(prediction.path)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1 * prediction.orig_shape[0]/1300
    font_thickness = int(prediction.orig_shape[0]/1000)
    color = [(128, 0, 0), (154, 99, 36), (70, 153, 144), (0, 0, 117), (230, 25, 75), (245, 130, 49), (255, 225, 25), (60, 180, 75), (66, 212, 244), (67, 99, 216), (240, 50, 230)]


    for bbox, label, confidence in zip(prediction.boxes.xyxy.numpy(), prediction.boxes.cls.numpy(), prediction.boxes.conf.numpy()):
        bbox = [ int(x) for x in bbox ]
        left, top, right, bottom = bbox
        cv2.rectangle(image, (left, top), (right, bottom), color[int(label)], font_thickness)
        

    for bbox, label, confidence in zip(prediction.boxes.xyxy.numpy(), prediction.boxes.cls.numpy(), prediction.boxes.conf.numpy()):
        bbox = [ int(x) for x in bbox ]
        left, top, right, bottom = bbox
        text = f"{prediction.names[label]}: {confidence:.2f}"
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        cv2.rectangle(image, (left, top - 10 - text_size[1]), (left + text_size[0], top - 10), color[int(label)], -1)
        cv2.putText(image, text, (left, top - 10), font, font_scale, (255, 255, 255), font_thickness)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(OUTPUT_DIR + prediction.path.split("/")[-1], image)

    # cv2.imshow('Predictions', image)

if __name__ == "__main__":
    predict_model()