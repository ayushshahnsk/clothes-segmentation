import cv2
import os
import numpy as np

# ================= PATHS =================
image_dir = r"C:\Users\ayush\Downloads\archive\Virtual tryon data\train\cloth"
mask_dir = "outputs"                 # predicted masks (.png)
extracted_dir = "outputs_extracted"  # extracted cloth
output_dir = "outputs_compare"       # final comparison

os.makedirs(output_dir, exist_ok=True)

count = 0

# ================= PROCESS =================
for img_name in os.listdir(image_dir):
    if not img_name.lower().endswith(".jpg"):
        continue

    img_path = os.path.join(image_dir, img_name)
    mask_path = os.path.join(mask_dir, img_name.replace(".jpg", ".png"))
    extracted_path = os.path.join(extracted_dir, img_name)

    if not os.path.exists(mask_path) or not os.path.exists(extracted_path):
        continue

    # Read images
    original = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    extracted = cv2.imread(extracted_path)

    if original is None or mask is None or extracted is None:
        continue

    # Resize all to same size
    h, w = 256, 256
    original = cv2.resize(original, (w, h))
    extracted = cv2.resize(extracted, (w, h))
    mask = cv2.resize(mask, (w, h))

    # Convert mask to 3-channel
    mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Stack images side by side
    combined = np.hstack((original, mask_color, extracted))

    # Save
    out_name = img_name.replace(".jpg", "_compare.jpg")
    cv2.imwrite(os.path.join(output_dir, out_name), combined)

    count += 1

print(f"✅ {count} comparison images saved in '{output_dir}'")
