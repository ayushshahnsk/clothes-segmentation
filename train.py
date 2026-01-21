import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from dataset import ClothesDataset
from model import UNet

dataset = ClothesDataset(
    cloth_dir=r"C:\Users\ayush\Downloads\archive\Virtual tryon data\train\cloth",
    mask_dir=r"C:\Users\ayush\Downloads\archive\Virtual tryon data\train\cloth-mask",
)

loader = DataLoader(dataset, batch_size=4, shuffle=True)

model = UNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.BCEWithLogitsLoss()

for epoch in range(5):
    for cloths, masks in loader:
        preds = model(cloths)
        loss = loss_fn(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

torch.save(model.state_dict(), "cloth_seg_model.pth")
print("Model saved as cloth_seg_model.pth")
