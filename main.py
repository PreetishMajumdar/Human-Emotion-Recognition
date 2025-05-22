import cv2
from ultralytics import YOLO
from threading import Thread

# Load your trained YOLOv8 model
model = YOLO('runs/detect/emotion_yolov8n/weights/best.pt')  # Trained on happy, sad, angry

# Print class labels
print("Loaded classes:", model.names)  # Should show: {0: 'happy', 1: 'sad', 2: 'angry'}

# Threaded video stream for better FPS
class VideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            if not self.stream.isOpened():
                continue
            self.grabbed, self.frame = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

# Start the threaded video stream
vs = VideoStream(src=1).start()  # Change src=1 if 0 doesn't work
cv2.namedWindow("Emotion Detection", cv2.WINDOW_NORMAL)

while True:
    frame = vs.read()
    if frame is None:
        continue

    # Resize and convert to RGB
    resized = cv2.resize(frame, (640, 640))
    rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # Run YOLOv8 prediction
    results = model(rgb_image, verbose=False)[0]

    # Draw bounding boxes and labels
    if results.boxes is not None:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            label = model.names.get(cls_id, "Unknown")
            conf = float(box.conf[0])

            # Draw rectangle and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f'{label} ({conf:.2f})', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    # Show the frame
    cv2.imshow("Emotion Detection", frame)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Clean up
vs.stop()
cv2.destroyAllWindows()
