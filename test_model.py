import torch
import cv2
import matplotlib.pyplot as plt
from model import UNet

# Load model
model = UNet()
model.load_state_dict(torch.load("cloth_seg_model.pth"))
model.eval()

# Change this image name if needed
# img_path = r"C:\Users\ayush\Downloads\archive\Virtual tryon data\train\cloth\14634_00.jpg"
# img_path = r"C:\Users\ayush\Downloads\archive\Virtual tryon data\train\cloth\14640_00.jpg"
img_path = r"C:\Users\ayush\Downloads\archive\Virtual tryon data\train\cloth\14666_00.jpg"
# img_path = r"C:\Users\ayush\Downloads\archive\Virtual tryon data\train\cloth\14485_00.jpg"
# img_path = r"C:\Users\ayush\Downloads\archive\Virtual tryon data\train\cloth\14500_00.jpg"

img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (256, 256))

img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0) / 255.0

with torch.no_grad():
    pred = model(img_tensor)

pred_mask = pred.squeeze().numpy()

plt.subplot(1, 2, 1)
plt.title("Cloth Image")
plt.imshow(img)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Predicted Mask")
plt.imshow(pred_mask, cmap="gray")
plt.axis("off")

plt.show()
