import tensorflow as tf

def load_fer2013(batch_size=64):
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fer2013.load_data()
    train_images = tf.expand_dims(train_images, -1) / 255.0
    test_images = tf.expand_dims(test_images, -1) / 255.0
    return tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(batch_size)
import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize and convert to RGB for YOLO
    resized = cv2.resize(frame, (640, 640))
    rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # YOLO expects RGB image
    results = model(rgb_image)

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, model.names[cls], (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

from threading import Thread

class VideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False
    
    def start(self):
        Thread(target=self.update, args=()).start()
        return self
    
    def update(self):
        while not self.stopped:
            self.grabbed, self.frame = self.stream.read()
    
    def read(self):
        return self.frame
    
    def stop(self):
        self.stopped = True