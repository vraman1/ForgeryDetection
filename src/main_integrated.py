import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from infer import load_model, predict_image_with_localization
from sift_analyzer import enhanced_sift_analysis
from copy_move_detector import detect_copy_move

# ---------------- CONFIG ---------------- #
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))   # …/project/src
ROOT_DIR = os.path.dirname(CURRENT_DIR)                    # …/project
MODEL_PATH = os.path.join(ROOT_DIR, "saved_models", "best_hybrid_classifier.pth")
# MODEL_PATH = "../saved_models/best_hybrid_classifier.pth"

model = load_model(MODEL_PATH, DEVICE)
#os.makedirs("final_results", exist_ok=True)
os.makedirs("../results", exist_ok=True)


def run_forgery_detection(image_path):
    image_name = os.path.basename(image_path)
    name_no_ext = os.path.splitext(image_name)[0]

    print("\nStarting Forgery Detection...\n")

    # -------- 1) CLASSIFIER + HEATMAP -------- #
    predicted_class, confidence, heatmap_image, _ = predict_image_with_localization(
        model, image_path, DEVICE
    )

    # Save heatmap overlay
    #heatmap_out = f"final_results/{name_no_ext}_heatmap.png"
    heatmap_out = f"../results/{name_no_ext}_heatmap.png"
    heatmap_bgr = cv2.cvtColor(np.array(heatmap_image), cv2.COLOR_RGB2BGR)
    cv2.imwrite(heatmap_out, heatmap_bgr)
    print(f"Heatmap saved → {heatmap_out}")

    status_text = "TAMPERED" if predicted_class == 1 else "AUTHENTIC"
    status_color = "red" if predicted_class == 1 else "green"

    # For display
    heat_rgb = np.array(heatmap_image)

    # -------- 2) SIFT -------- #
    print("\nRunning SIFT Keypoint Analysis...")
    kp_count, sift_saved_path, keypoints, descriptors, sift_stats = enhanced_sift_analysis(
        #image_path, output_dir="final_results"
        image_path, output_dir="../results"
    )

    #sift_out = f"final_results/{name_no_ext}_sift.jpg"
    sift_out = f"../results/{name_no_ext}_sift.jpg"
    try:
        os.rename(sift_saved_path, sift_out)
    except:
        sift_out = sift_saved_path

    sift_img = cv2.imread(sift_out)
    sift_img = cv2.cvtColor(sift_img, cv2.COLOR_BGR2RGB)
    print(f"SIFT result saved → {sift_out}")

    # -------- 3) COPY-MOVE -------- #
    print("\nRunning Copy-Move Forgery Detection...")
    #result = detect_copy_move(image_path, output_dir="final_results")
    result = detect_copy_move(image_path, output_dir="../results")

    # Handle both return types: (bool, path) or just bool
    if isinstance(result, tuple):
        copy_success, copy_move_path = result
    else:
        copy_success = result
        copy_move_path = None

    if copy_success and copy_move_path:
        # Read and process copy-move image to make lines thinner and dots smaller
        cm_img = cv2.imread(copy_move_path)
        if cm_img is not None:
            cm_img = cv2.cvtColor(cm_img, cv2.COLOR_BGR2RGB)
            
            # Resize down and up to make lines thinner
            h, w = cm_img.shape[:2]
            # Resize to half size and back to original to thin lines
            cm_img_small = cv2.resize(cm_img, (w//2, h//2), interpolation=cv2.INTER_AREA)
            cm_img = cv2.resize(cm_img_small, (w, h), interpolation=cv2.INTER_CUBIC)
            
            print(f"Copy-Move result saved → {copy_move_path}")
        else:
            cm_img = None
            print("❌ Failed to load copy-move result image")
    else:
        cm_img = None
        print("⚠️ No Copy-Move forgery detected.")

    # -------- EXACT DISPLAY LIKE REFERENCE -------- #
    orig = cv2.imread(image_path)
    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)

    # ADD RESIZE HERE:
    target_width = 800
    height, width = orig.shape[:2]
    scale = target_width / width
    new_height = int(height * scale)
    orig = cv2.resize(orig, (target_width, new_height))

    # Also resize the heatmap to match:
    heat_rgb = cv2.resize(heat_rgb, (target_width, new_height))

    # Create display exactly like reference - 2x2 layout with bigger size
    plt.figure(figsize=(24, 18))
    #plt.subplots_adjust(top=0.85, bottom=0.15, left=0.05, right=0.95,
      #              hspace=0.4, wspace=0.4)  # Even more space
    plt.tight_layout(pad=0.5, h_pad=1.0, w_pad=1.0)
    
    # Main title at the top
    plt.suptitle("FORGERY IMAGE DETECTION", fontsize=20, fontweight='bold', y=0.95, color='#8B0000')

    # 1. ORIGINAL IMAGE (Top Left)
    plt.subplot(2, 2, 1)
    plt.imshow(orig)
    plt.title("Original Image", fontsize=11, fontweight='bold', pad=5)
    plt.axis("off")

    # 2. BINARY CLASSIFIER RESULT (Top Right)
    plt.subplot(2, 2, 2)
    plt.imshow(heat_rgb)
    plt.title("Binary Classifier", fontsize=11, fontweight='bold', pad=5)
    
    # Add the AUTHENTIC/TAMPERED text on the image (no blue box)
    # plt.text(0.5, 0.95, f"{status_text} ({confidence*100:.2f}%)", 
             #fontsize=20, fontweight='bold', ha='center', va='top',
             #transform=plt.gca().transAxes, color=status_color)
    plt.axis("off")

    # 3. SIFT KEYPOINTS (Bottom Left)
    plt.subplot(2, 2, 3)
    plt.imshow(sift_img)
    plt.title(f"SIFT Keypoints ({kp_count})", fontsize=11, fontweight='bold', pad=10)
    plt.axis("off")

    # 4. COPY-MOVE DETECTION (Bottom Right)
    plt.subplot(2, 2, 4)
    if cm_img is not None:
        plt.imshow(cm_img)
        plt.title("Copy-Move Detect", fontsize=11, fontweight='bold', pad=10)
    else:
        # Create blank image with the text
        h, w = orig.shape[:2]
        blank_bg = np.ones((h, w, 3)) * 240  # Light gray background
        plt.imshow(blank_bg.astype(np.uint8))
        plt.text(0.5, 0.5, "NO COPY-MOVE\nDETECTED", 
                 fontsize=11, fontweight='bold', ha='center', va='center',
                 transform=plt.gca().transAxes, color='black',
                 linespacing=1.5)
        plt.title("Copy-Move Detect", fontsize=11, fontweight='bold', pad=10)
    plt.axis("off")

    # plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    #print("\n🎯 All results saved in → final_results/\n")
    print("\nAll results saved in → results/\n")


# ---------------- MAIN ---------------- #
if __name__ == "__main__":
    print("\n🔎 Forgery Detection System Ready.\n")
    while True:
        image_path = input("Enter image path (or 'quit'): ")
        if image_path.lower() == "quit":
            print("\nExiting...")
            break
        if not os.path.exists(image_path):
            print("❌ File not found, try again.\n")
            continue
        run_forgery_detection(image_path)