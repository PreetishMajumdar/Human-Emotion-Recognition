import cv2

cap = cv2.VideoCapture(0)  # Try changing 0 to 1 or 2 if needed

if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break
    cv2.imshow('Test Webcam', frame)
    if cv2.waitKey(1) == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
