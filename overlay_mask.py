import cv2
import os

# ================== PATHS ==================

# Original cloth images
image_dir = r"C:\Users\ayush\Downloads\archive\Virtual tryon data\train\cloth"

# Folder where predicted masks are ACTUALLY saved
# ⚠️ CHANGE THIS if your masks are elsewhere
mask_dir = r"outputs"  # example: outputs / results / predicted_masks

# Output folder
output_dir = "outputs_extracted"

os.makedirs(output_dir, exist_ok=True)

# ================== PROCESS ==================

count = 0
total = 0

print("🔍 Image directory:", image_dir)
print("🔍 Mask directory:", mask_dir)
print("📂 Output directory:", output_dir)
print("-" * 50)

for img_name in os.listdir(image_dir):
    if not img_name.lower().endswith(".jpg"):
        continue

    total += 1

    img_path = os.path.join(image_dir, img_name)

    # Try common mask name patterns
    possible_masks = [
        img_name.replace(".jpg", ".png"),
        img_name.replace(".jpg", "_mask.png"),
        img_name.replace(".jpg", ".jpg"),
    ]

    mask_path = None
    for m in possible_masks:
        p = os.path.join(mask_dir, m)
        if os.path.exists(p):
            mask_path = p
            break

    print(f"\n🖼 Image: {img_name}")

    if mask_path is None:
        print("❌ Mask NOT found for this image")
        continue

    print("✅ Mask found:", os.path.basename(mask_path))

    image = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        print("❌ Failed to read image or mask")
        continue

    # Resize mask to match image
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    # Ensure binary mask
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Apply mask
    extracted = cv2.bitwise_and(image, image, mask=mask)

    # Save output
    out_path = os.path.join(output_dir, img_name)
    cv2.imwrite(out_path, extracted)

    print("💾 Saved:", out_path)
    count += 1

print("\n================== SUMMARY ==================")
print(f"Total cloth images checked : {total}")
print(f"Images successfully extracted : {count}")
print("============================================")
