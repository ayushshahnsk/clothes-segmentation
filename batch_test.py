import os
import torch
import cv2
from model import UNet

model = UNet()
model.load_state_dict(torch.load("cloth_seg_model.pth"))
model.eval()

cloth_dir = r"C:\Users\ayush\Downloads\archive\Virtual tryon data\train\cloth"
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

for img_name in os.listdir(cloth_dir):
    img_path = os.path.join(cloth_dir, img_name)
    img = cv2.imread(img_path)
    img = cv2.resize(img, (256, 256))

    img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0) / 255.0

    with torch.no_grad():
        pred = model(img_tensor)

    mask = pred.squeeze().numpy() * 255
    cv2.imwrite(os.path.join(output_dir, img_name), mask)

print("All masks generated in 'outputs' folder")
