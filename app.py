import streamlit as st
import torch
import cv2
import numpy as np
from model import UNet
import matplotlib.pyplot as plt
import os
import shutil


# ================= UI CONFIG =================
st.set_page_config(page_title="Cloth Segmentation UI", layout="wide")
st.title("👕 Clothes Segmentation - Smart Dashboard")


# ================= LOAD MODEL =================
@st.cache_resource
def load_ui_model():
    model = UNet()
    model.load_state_dict(torch.load("cloth_seg_model.pth", map_location="cpu"))
    model.eval()
    return model


model = load_ui_model()
st.success("Model loaded successfully!")

# ================= MODE SELECT =================
mode = st.radio(
    "Select Mode",
    ["Single Image", "Multi Image Dashboard", "Virtual Try-On"],
    horizontal=True,
)

# ================= SIDEBAR CONTROLS =================
st.sidebar.header("⚙ Controls")
threshold = st.sidebar.slider("Mask Threshold", 0.1, 0.9, 0.5, 0.05)
show_stats = st.sidebar.checkbox("Show Pixel Stats", True)


# ================= UTILS =================
def to_bytes(img):
    _, buffer = cv2.imencode(".png", img)
    return buffer.tobytes()


# ================= SINGLE IMAGE MODE =================
if mode == "Single Image":

    uploaded_file = st.file_uploader(
        "Upload a cloth image", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            st.error("Invalid image!")
        else:
            original = img.copy()

            img = cv2.resize(img, (256, 256))
            img = img.astype("float32") / 255.0
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

            if st.button("🚀 Run Segmentation"):
                with st.spinner("Processing..."):
                    with torch.no_grad():
                        pred = model(img_tensor)

                    mask = pred.squeeze().numpy()
                    mask = (mask > threshold).astype("uint8") * 255

                    mask_resized = cv2.resize(
                        mask, (original.shape[1], original.shape[0])
                    )
                    extracted = cv2.bitwise_and(original, original, mask=mask_resized)

                    mask_color = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)
                    side_by_side = np.hstack((original, mask_color, extracted))

                # -------- DISPLAY --------
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.subheader("Original")
                    st.image(
                        cv2.cvtColor(original, cv2.COLOR_BGR2RGB),
                        use_container_width=True,
                    )

                with col2:
                    st.subheader("Mask")
                    st.image(mask_resized, clamp=True)

                with col3:
                    st.subheader("Extracted Cloth")
                    st.image(
                        cv2.cvtColor(extracted, cv2.COLOR_BGR2RGB),
                        use_container_width=True,
                    )

                st.subheader("Side by Side Comparison")
                st.image(
                    cv2.cvtColor(side_by_side, cv2.COLOR_BGR2RGB),
                    use_container_width=True,
                )

                # -------- DOWNLOAD --------
                st.subheader("📥 Download Results")
                col4, col5, col6 = st.columns(3)

                with col4:
                    st.download_button(
                        "Download Mask", to_bytes(mask_resized), file_name="mask.png"
                    )
                with col5:
                    st.download_button(
                        "Download Extracted",
                        to_bytes(extracted),
                        file_name="extracted.png",
                    )
                with col6:
                    st.download_button(
                        "Download Compare",
                        to_bytes(side_by_side),
                        file_name="comparison.png",
                    )

                # -------- STATS --------
                if show_stats:
                    total_pixels = mask_resized.size
                    cloth_pixels = np.count_nonzero(mask_resized)
                    percentage = (cloth_pixels / total_pixels) * 100

                    st.subheader("📊 Pixel Statistics")
                    st.write(f"Cloth Area: {percentage:.2f}% of image")
                    st.write(f"Cloth Pixels: {cloth_pixels:,} px")
                    st.write(f"Total Pixels: {total_pixels:,} px")

                # -------- GRAPH ANALYTICS (FIXED) --------
                st.subheader("📈 Graph Analytics")
                bg_pixels = mask_resized.size - cloth_pixels

                chart_col1, chart_col2 = st.columns(2)

                with chart_col1:
                    st.markdown("### Cloth vs Background")
                    st.bar_chart(
                        {
                            "Cloth Pixels (px)": cloth_pixels,
                            "Background Pixels (px)": bg_pixels,
                        }
                    )

                with chart_col2:
                    st.markdown("### Pixel Distribution")
                    st.json(
                        {
                            "Cloth Pixels (px)": int(cloth_pixels),
                            "Background Pixels (px)": int(bg_pixels),
                            "Total Pixels (px)": int(mask_resized.size),
                        }
                    )

# ================= MULTI IMAGE DASHBOARD =================
if mode == "Multi Image Dashboard":

    uploaded_files = st.file_uploader(
        "Upload multiple cloth images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if uploaded_files:

        st.subheader("📂 Multi Image Dashboard")

        summary = []

        for uploaded_file in uploaded_files:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img is None:
                continue

            original = img.copy()
            img = cv2.resize(img, (256, 256))
            img = img.astype("float32") / 255.0
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

            with torch.no_grad():
                pred = model(img_tensor)

            mask = pred.squeeze().numpy()
            mask = (mask > threshold).astype("uint8") * 255
            mask_resized = cv2.resize(mask, (original.shape[1], original.shape[0]))
            extracted = cv2.bitwise_and(original, original, mask=mask_resized)

            cloth_pixels = np.count_nonzero(mask_resized)
            total_pixels = mask_resized.size
            percentage = (cloth_pixels / total_pixels) * 100

            summary.append(
                (uploaded_file.name, cloth_pixels, percentage, original, extracted)
            )

        st.subheader("📊 Summary Table")
        st.table(
            [
                {
                    "Image": s[0],
                    "Cloth Pixels (px)": s[1],
                    "Cloth Area (%)": round(s[2], 2),
                }
                for s in summary
            ]
        )

        st.subheader("🖼 Image Grid")
        cols = st.columns(3)

        for idx, item in enumerate(summary):
            with cols[idx % 3]:
                st.image(
                    cv2.cvtColor(item[3], cv2.COLOR_BGR2RGB),
                    caption=item[0],
                    use_container_width=True,
                )
                st.image(
                    cv2.cvtColor(item[4], cv2.COLOR_BGR2RGB),
                    caption="Extracted",
                    use_container_width=True,
                )

# ================= VIRTUAL TRY-ON (DATASET MODE) =================
if mode == "Virtual Try-On":

    st.subheader("👗 AI Virtual Try-On")

    dataset_path = "VITON-HD/datasets/test"
    image_dir = os.path.join(dataset_path, "image")
    cloth_dir = os.path.join(dataset_path, "cloth")

    humans = sorted([f for f in os.listdir(image_dir) if f.endswith(".jpg")])
    clothes = sorted([f for f in os.listdir(cloth_dir) if f.endswith(".jpg")])

    col1, col2 = st.columns(2)

    with col1:
        human_img = st.selectbox("Select Human Image", humans)

    with col2:
        cloth_img = st.selectbox("Select Cloth Image", clothes)

    if human_img and cloth_img:

        st.image(
            os.path.join(image_dir, human_img), caption="Selected Human", width=250
        )
        st.image(
            os.path.join(cloth_dir, cloth_img), caption="Selected Cloth", width=250
        )

        if st.button("✨ Try Cloth On"):

            with st.spinner("Running AI Virtual Try-On... Please wait ⏳"):

                # Write test_pairs.txt
                pairs_path = "VITON-HD/datasets/test/test_pairs.txt"
                with open(pairs_path, "w") as f:
                    f.write(f"{human_img} {cloth_img}")

                # Clear old result
                result_dir = "VITON-HD/results/streamlit_tryon"
                if os.path.exists(result_dir):
                    shutil.rmtree(result_dir)

                # Run VITON-HD
                cmd = (
                    "python VITON-HD/test.py "
                    "--name streamlit_tryon "
                    "--dataset_dir VITON-HD/datasets "
                    "--dataset_mode test "
                    "--dataset_list test/test_pairs.txt "
                    "--checkpoint_dir VITON-HD/checkpoints "
                    "--save_dir VITON-HD/results"
                )

                ret = os.system(cmd)
                st.write("Return Code:", ret)

                # Load result
                result_path = "VITON-HD/results/streamlit_tryon"
                all_imgs = []
                for root, dirs, files in os.walk(result_path):
                    for file in files:
                        if file.endswith(".jpg") or file.endswith(".png"):
                            all_imgs.append(os.path.join(root, file))

                if len(all_imgs) > 0:
                    result_img = cv2.imread(all_imgs[0])
                    st.subheader("🎉 Try-On Result")
                    st.image(
                        cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB),
                        use_container_width=True,
                    )
                else:
                    st.error("❌ No result generated. Check VITON-HD logs.")
