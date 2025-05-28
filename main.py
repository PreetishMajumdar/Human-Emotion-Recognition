import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("Models/emotion.h5")

# Emotion labels and corresponding emoji characters
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
emoticon_chars = {
    'angry': '😠',
    'disgust': '🤢',
    'fear': '😨',
    'happy': '😊',
    'neutral': '😐',
    'sad': '😢',
    'surprise': '😲'
}

# Load emoji PNG images with alpha channel (make sure files exist)
emoji_imgs = {
    emotion: cv2.imread(f"Faces/{emotion}.png", cv2.IMREAD_UNCHANGED)
    for emotion in emotion_labels
}

def overlay_image_alpha(img, img_overlay, x, y, alpha_mask):
    """Overlay img_overlay on top of img at (x, y) with alpha_mask."""
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    alpha = alpha_mask[y1o:y2o, x1o:x2o, None] / 255.0
    img[y1:y2, x1:x2] = (1. - alpha) * img[y1:y2, x1:x2] + alpha * img_overlay[y1o:y2o, x1o:x2o]

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

img_size = 48
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    frame_emoticons = []

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_resized = cv2.resize(roi_gray, (img_size, img_size))
        roi_normalized = roi_resized / 255.0
        roi_reshaped = np.reshape(roi_normalized, (1, img_size, img_size, 1))

        prediction = model.predict(roi_reshaped, verbose=0)
        emotion_idx = np.argmax(prediction)
        emotion_text = emotion_labels[emotion_idx]
        emoticon = emoticon_chars[emotion_text]
        frame_emoticons.append(emoticon)

        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Overlay emoji image ABOVE the bounding box (not inside)
        emoji_img = emoji_imgs.get(emotion_text)
        if emoji_img is not None:
            scale = 1.0  # adjust emoji size relative to face width
            emoji_width = int(w * scale)
            emoji_height = int(emoji_img.shape[0] * (emoji_width / emoji_img.shape[1]))
            emoji_resized = cv2.resize(emoji_img, (emoji_width, emoji_height), interpolation=cv2.INTER_AREA)

            overlay_x = x
            overlay_y = y - emoji_height - 5  # 5 pixels above the box

            # If going above the frame, place emoji just inside frame top
            if overlay_y < 0:
                overlay_y = 0

            overlay_image_alpha(frame, emoji_resized[:, :, :3], overlay_x, overlay_y, emoji_resized[:, :, 3])

    frame_count += 1
    if frame_count % 100 == 0 and frame_emoticons:
        with open("logs/emotions_log.txt", "a", encoding="utf-8") as file:
            file.write(" ".join(frame_emoticons) + "\n")
        print(f"Logged emoticons at frame {frame_count}: {' '.join(frame_emoticons)}")

    cv2.imshow("Emotion Detection with Emojis", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
