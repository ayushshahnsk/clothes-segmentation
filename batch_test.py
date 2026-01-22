import os
import torch
import cv2
import numpy as np
from model import UNet

# ------------------ MODEL ------------------
model = UNet()
model.load_state_dict(torch.load("cloth_seg_model.pth", map_location="cpu"))
model.eval()

# ------------------ PATHS ------------------
cloth_dir = r"C:\Users\ayush\Downloads\archive\Virtual tryon data\train\cloth"
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

count = 0

# ------------------ PROCESS ----------------
for img_name in os.listdir(cloth_dir):
    if not img_name.lower().endswith(".jpg"):
        continue

    img_path = os.path.join(cloth_dir, img_name)

    img = cv2.imread(img_path)
    if img is None:
        continue

    img = cv2.resize(img, (256, 256))
    img = img.astype("float32") / 255.0

    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        pred = model(img_tensor)

    # Convert prediction to binary mask
    mask = pred.squeeze().numpy()
    mask = (mask > 0.5).astype("uint8") * 255

    # Save as PNG (IMPORTANT)
    mask_name = img_name.replace(".jpg", ".png")
    cv2.imwrite(os.path.join(output_dir, mask_name), mask)

    count += 1

print(f"✅ {count} masks saved in '{output_dir}' folder")
