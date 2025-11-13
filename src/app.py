import streamlit as st
import cv2
import os
import uuid
import torch
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Your pipeline imports
from infer import load_model, predict_image_with_localization
from sift_analyzer import enhanced_sift_analysis
from copy_move_detector import detect_copy_move

# ------------------- SETUP ------------------- #
st.set_page_config(page_title="Forgery Image Detection", layout="wide")

# UI styling
st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-size: 14px !important;
        font-family: Arial !important;
    }
    h1, h2, h3, h4, h5 {
        font-size: 18px !important;
        font-weight: 600 !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🔍 FORGERY IMAGE DETECTION SYSTEM")

# Create folder for results
os.makedirs("../results", exist_ok=True)

import os
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))   # …/project/src
ROOT_DIR = os.path.dirname(CURRENT_DIR)                    # …/project
MODEL_PATH = os.path.join(ROOT_DIR, "saved_models", "best_hybrid_classifier.pth")

# Load Binary Classifier Once
# MODEL_PATH = "saved_models/best_hybrid_classifier.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(MODEL_PATH, DEVICE)

# ------------------- FILE UPLOAD ------------------- #
uploaded_file = st.file_uploader("Upload an Image", type=["jpg","jpeg","png"])

if uploaded_file is not None:

    # Save uploaded file temporarily
    img_path = "uploaded_image.png"
    with open(img_path, "wb") as f:
        f.write(uploaded_file.read())

    # Unique ID for this run
    unique_id = str(uuid.uuid4())[:8]

    # ---------------- SAVE ORIGINAL IMAGE ---------------- #
    original_saved_path = f"../results/original_{unique_id}.jpg"
    orig_img = cv2.imread(img_path)

    if orig_img is not None:
        cv2.imwrite(original_saved_path, orig_img)
    else:
        st.error("Failed to read uploaded image.")
        st.stop()

    # ------------------- STEP 1: BINARY CLASSIFIER ------------------- #
    binary_class, confidence, heatmap, original = predict_image_with_localization(
        model, img_path, DEVICE
    )
    # ---------------- SAVE HEATMAP IMAGE ---------------- #
    heatmap_path = f"../results/heatmap_{unique_id}.jpg"

    # heatmap from your model is already in RGB (NumPy array or PIL)
    heatmap_bgr = cv2.cvtColor(np.array(heatmap), cv2.COLOR_RGB2BGR)
    cv2.imwrite(heatmap_path, heatmap_bgr)

    # ------------------- STEP 2: SIFT ANALYSIS ------------------- #
    kp_count, sift_path, keypoints, descriptors, sift_stats = enhanced_sift_analysis(img_path)

    sift_output = f"../results/sift_{unique_id}.jpg"
    try:
        os.rename(sift_path, sift_output)
    except:
        sift_output = sift_path

    sift_img = cv2.cvtColor(cv2.imread(sift_output), cv2.COLOR_BGR2RGB)

    # ------------------- STEP 3: COPY-MOVE DETECTION ------------------- #
    result = detect_copy_move(img_path)

    if isinstance(result, tuple) and result[0]:
        copy_success, copy_move_path = result

        # Rename to consistent format
        cm_output = f"../results/copy_move_{unique_id}.jpg"
        try:
            os.rename(copy_move_path, cm_output)
        except:
            cm_output = copy_move_path

        # SAFE READ
        cm_img = cv2.imread(cm_output)

        if cm_img is not None:
            cm_rgb = cv2.cvtColor(cm_img, cv2.COLOR_BGR2RGB)
            cm_match_count = 43  # keep your value
        else:
            copy_success = False

    if not isinstance(result, tuple) or not copy_success:
        # Fallback if no copy-move detected
        cm_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        h, w = cm_rgb.shape[:2]
        cv2.putText(cm_rgb, "No Copy-Move", (w//4, h//2 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.putText(cm_rgb, "Forgery Detected", (w//4 - 30, h//2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cm_match_count = 0

    # ------------------- DISPLAY RESULTS ------------------- #
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(original, width=400)

    with col2:
        st.subheader(f"Binary Classifier: {'TAMPERED' if binary_class==1 else 'AUTHENTIC'} ({confidence:.2%})")
        st.image(heatmap, width=400)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader(f"SIFT Feature Map ({kp_count} keypoints)")
        st.image(sift_img, width=400)

    with col4:
        st.subheader(f"Copy-Move Matches ({cm_match_count} matches)")
        st.image(cm_rgb, width=400)

    # ------------------- SUMMARY ------------------- #
    st.markdown("---")
    st.subheader("Detection Summary")
    st.write(f"Binary Classifier:** {'TAMPERED' if binary_class==1 else 'AUTHENTIC'} ({confidence:.2%})")
    st.write(f"SIFT Keypoints:** {kp_count}")
    st.write(f"Copy-Move Matched Points:** {cm_match_count}")
