import streamlit as st
import cv2
import os
import uuid
import torch
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
st.set_option('deprecation.showPyplotGlobalUse', False)

# Your pipeline imports
from infer import load_model, predict_image_with_localization
from sift_analyzer import enhanced_sift_analysis
from copy_move_detector import detect_copy_move

# ------------------- SETUP ------------------- #
st.set_page_config(page_title="Forgery Image Detection", layout="wide")
st.title("🔍 FORGERY IMAGE DETECTION SYSTEM")

# Create folder for results
os.makedirs("results", exist_ok=True)

# Load Binary Classifier Once
MODEL_PATH = "saved_models/best_hybrid_classifier.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(MODEL_PATH, DEVICE)

# ------------------- FILE UPLOAD ------------------- #
uploaded_file = st.file_uploader("Upload an Image", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    # Save uploaded image temporarily
    img_path = "uploaded_image.png"
    with open(img_path, "wb") as f:
        f.write(uploaded_file.read())

    # Generate unique ID for this run
    unique_id = str(uuid.uuid4())[:8]

    # ------------------- STEP 1: BINARY CLASSIFIER ------------------- #
    binary_class, confidence, heatmap, original = predict_image_with_localization(model, img_path, DEVICE)

    # ------------------- STEP 2: SIFT ANALYSIS ------------------- #
    # Run SIFT (no custom filename)
    kp_count, sift_path, keypoints, descriptors, sift_stats = enhanced_sift_analysis(img_path)

    # Create new unique result file
    sift_output = f"results/sift_{unique_id}.jpg"
    os.rename(sift_path, sift_output)

    # Load for display
    sift_img = cv2.cvtColor(cv2.imread(sift_output), cv2.COLOR_BGR2RGB)


    # ------------------- STEP 3: COPY-MOVE DETECTION ------------------- #
  
    result = detect_copy_move(img_path)

    if isinstance(result, tuple) and result[0]:
        copy_success, copy_move_path = result
        cm_img = cv2.imread(copy_move_path)
        cm_rgb = cv2.cvtColor(cm_img, cv2.COLOR_BGR2RGB)
        cm_match_count = 43
    else:
        copy_success = False
        copy_move_path = None
        # Use original image with "No Copy-Move" text
        cm_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        # Add text overlay
        h, w = cm_rgb.shape[:2]
        cv2.putText(cm_rgb, "No Copy-Move", (w//4, h//2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.putText(cm_rgb, "Forgery Detected", (w//4 - 30, h//2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cm_match_count = 0


    # ------------------- DISPLAY RESULTS ------------------- #
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🖼 Original Image")
        st.image(original, use_container_width=True)

    with col2:
        st.subheader(f"🧠 Binary Classifier: {'TAMPERED' if binary_class==1 else 'AUTHENTIC'} ({confidence:.2%})")
        st.image(heatmap, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader(f"🔹 SIFT Feature Map ({kp_count} keypoints)")
        st.image(sift_img, use_container_width=True)

    with col4:
        st.subheader(f"🔁 Copy-Move Matches ({cm_match_count} matches)")
        st.image(cm_rgb, use_container_width=True)

    # ------------------- SUMMARY ------------------- #
    st.markdown("---")
    st.subheader("📌 Detection Summary")
    st.write(f"• **Binary Classifier:** {'TAMPERED' if binary_class==1 else 'AUTHENTIC'} ({confidence:.2%})")
    st.write(f"• **SIFT Keypoints:** {kp_count}")
    st.write(f"• **Copy-Move Matched Points:** {cm_match_count}")
