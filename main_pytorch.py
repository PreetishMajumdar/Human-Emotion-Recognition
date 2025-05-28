import torch
import torch.nn as nn
import timm
import cv2
from torchvision import transforms
from PIL import Image

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Definition (from your notebook)
num_classes = 7
model = timm.create_model('xception', pretrained=False)
model.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
model.fc = nn.Linear(model.get_classifier().in_features, num_classes)
model.load_state_dict(torch.load("Models/xception_emotion_model_v3.pth", map_location=device))  # Update if needed
model.to(device)
model.eval()

# Preprocessing for grayscale 48x48 input
transform = transforms.Compose([
    transforms.Grayscale(),  # Webcam is color by default
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to PIL for transform
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).item()

    # Show prediction
    cv2.putText(frame, f'Predicted Class: {pred}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Live Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
