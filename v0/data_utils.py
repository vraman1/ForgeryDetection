import os
import cv2
import numpy as np
from tqdm import tqdm
import joblib
import random
import torch
from torch.utils.data import Dataset


# ==============================
# 📦 CONFIGURATION
# ==============================
PATCH_SIZE = 96
SIFT_FEATURES = 500  # balanced (400–800 recommended)
CONTRAST_THRESHOLD = 0.02
EDGE_THRESHOLD = 15
ENTROPY_RADIUS = 3


# ==============================
# 🔹 UTIL FUNCTIONS
# ==============================
def load_casia_dataset(base_path):
    """
    Loads CASIA2 dataset paths with correct GT file matching and caching.
    """
    # Cache file to avoid reloading on every run
    cache_file = os.path.join(base_path, "dataset_cache.pkl")
    
    # Check if cached version exists
    if os.path.exists(cache_file):
        print(f"📂 Loading cached dataset...")
        try:
            authentic_paths, tampered_paths, gt_mapping = joblib.load(cache_file)
            print(f"✅ Loaded from cache: {len(authentic_paths)} authentic, {len(tampered_paths)} tampered, {len(gt_mapping)} GT mappings")
            return authentic_paths, tampered_paths, gt_mapping
        except Exception as e:
            print(f"❌ Cache loading failed: {e}. Regenerating dataset...")
    
    authentic_dir = os.path.join(base_path, "Au")
    tampered_dir = os.path.join(base_path, "Tp")
    gt_dir = os.path.join(base_path, "CASIA 2 Groundtruth")

    # Load image paths
    authentic_paths = [os.path.join(authentic_dir, f) for f in os.listdir(authentic_dir) if f.lower().endswith((".jpg", ".png", ".jpeg", ".bmp"))]
    tampered_paths = [os.path.join(tampered_dir, f) for f in os.listdir(tampered_dir) if f.lower().endswith((".jpg", ".png", ".jpeg", ".bmp"))]

    gt_mapping = {}
    
    print(f"🔍 Looking for GT files in: {gt_dir}")
    
    if not os.path.exists(gt_dir):
        print(f"❌ GT directory not found: {gt_dir}")
        # Still return the results but with empty gt_mapping
        result = (authentic_paths, tampered_paths, gt_mapping)
        joblib.dump(result, cache_file)
        return result

    # CASIA 2.0 specific matching: same name + "_gt.png"
    matched_count = 0
    for img_path in tampered_paths:
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # GT file follows pattern: {same_name}_gt.png
        gt_filename = f"{img_name}_gt.png"
        gt_path = os.path.join(gt_dir, gt_filename)
        
        if os.path.exists(gt_path):
            gt_mapping[img_path] = gt_path
            matched_count += 1
            if matched_count <= 3:  # Print first 3 matches
                print(f"✅ Matched: {os.path.basename(img_path)} -> {gt_filename}")
        else:
            # Try alternative extensions in case some GT files have different formats
            for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
                gt_filename_alt = f"{img_name}_gt{ext}"
                gt_path_alt = os.path.join(gt_dir, gt_filename_alt)
                if os.path.exists(gt_path_alt):
                    gt_mapping[img_path] = gt_path_alt
                    matched_count += 1
                    if matched_count <= 3:
                        print(f"✅ Matched: {os.path.basename(img_path)} -> {gt_filename_alt}")
                    break

    print(f"✅ Loaded {len(authentic_paths)} authentic and {len(tampered_paths)} tampered images.")
    print(f"✅ GT mappings found: {len(gt_mapping)}/{len(tampered_paths)}")
    
    # Fallback: If no GT files found, treat all Tp images as tampered
    if len(gt_mapping) == 0:
        print("⚠️ No GT files found. Using fallback: all Tp images treated as tampered.")
        for img_path in tampered_paths:
            gt_mapping[img_path] = "TAMPERED_NO_GT"
    
    # Cache the results for future runs
    result = (authentic_paths, tampered_paths, gt_mapping)
    try:
        joblib.dump(result, cache_file)
        print(f"💾 Cached dataset to: {cache_file}")
    except Exception as e:
        print(f"⚠️ Could not cache dataset: {e}")
    
    return result


def split_dataset(authentic_paths, tampered_paths, test_size=0.3):
    """
    Splits dataset into train/test sets.
    """
    random.shuffle(authentic_paths)
    random.shuffle(tampered_paths)
    n_test_auth = int(len(authentic_paths) * test_size)
    n_test_tamp = int(len(tampered_paths) * test_size)

    return {
        "train": authentic_paths[n_test_auth:] + tampered_paths[n_test_tamp:],
        "test": authentic_paths[:n_test_auth] + tampered_paths[:n_test_tamp]
    }


# ==============================
# 🔹 ENTROPY + SIFT EXTRACTION
# ==============================
def compute_entropy(img_gray, radius=ENTROPY_RADIUS):
    """
    Computes local entropy for grayscale image.
    """
    kernel_size = 2 * radius + 1
    hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
    prob = hist / np.sum(hist)
    entropy = -np.sum(prob * np.log2(prob + 1e-9))
    ent_img = cv2.blur(img_gray.astype(np.float32), (kernel_size, kernel_size))
    return ent_img / np.max(ent_img)


def extract_sift_on_entropy(img, entropy_radius=ENTROPY_RADIUS, max_kp=SIFT_FEATURES):
    """
    Extracts SIFT features from entropy-enhanced grayscale image.
    """
     # ⚡ ADD THESE 4 LINES ONLY:
    h, w = img.shape[:2]
    if h > 512 or w > 512:
        scale = 512 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ent_norm = compute_entropy(gray, entropy_radius)
    ent_norm = (255 * (ent_norm / np.max(ent_norm))).astype(np.uint8)

    sift = cv2.SIFT_create(nfeatures=max_kp, contrastThreshold=CONTRAST_THRESHOLD, edgeThreshold=EDGE_THRESHOLD)
    kps, descs = sift.detectAndCompute(ent_norm, None)

    if kps is None or len(kps) == 0:
        print("⚠️ No keypoints detected for image — skipping.")
        return [], None, gray, None, 1.0

    return kps, descs, gray, ent_norm, 1.0


# ==============================
# 🔹 PATCH EXTRACTION (safe)
# ==============================
def extract_patch(gray, kp, size=PATCH_SIZE):
    """
    Safely extracts a square patch around a keypoint.
    """
    x, y = int(kp.pt[0]), int(kp.pt[1])
    half = size // 2
    h, w = gray.shape

    # Clip coordinates to avoid out-of-bound errors
    x1, y1 = max(0, x - half), max(0, y - half)
    x2, y2 = min(w, x + half), min(h, y + half)

    patch = gray[y1:y2, x1:x2]

    # Skip if patch empty or too small
    if patch.size == 0 or patch.shape[0] < 5 or patch.shape[1] < 5:
        return None

    try:
        patch_resized = cv2.resize(patch, (size, size))
    except cv2.error:
        return None

    return patch_resized


# ==============================
# 🔹 PAIR GENERATION
# ==============================
def generate_pairs_from_split(split, gt_mapping, patch_size=PATCH_SIZE, max_pairs=5000):
    """
    Generates meaningful patch pairs for training with proper tampered/authentic labels.
    """
    pairs, labels = [], []
    
    # Separate authentic and tampered paths from the split
    authentic_paths = [p for p in split if p not in gt_mapping]
    tampered_paths = [p for p in split if p in gt_mapping]
    
    print(f"🔍 Split contains {len(tampered_paths)} tampered, {len(authentic_paths)} authentic images")
    
    # Process each image and extract patches
    all_patches = {}
    
    for idx, img_path in enumerate(tqdm(split, desc="🧩 Extracting patches")):
        if len(pairs) >= max_pairs:
            break
            
        img = cv2.imread(img_path)
        if img is None:
            continue

        kps, descs, gray, _, _ = extract_sift_on_entropy(img)
        if not kps:
            continue

        # Extract multiple patches from this image
        patches = []
        for kp in kps[:10]:  # Take up to 10 patches per image
            patch = extract_patch(gray, kp, patch_size)
            if patch is not None:
                patches.append(patch)
            if len(patches) >= 5:  # Limit to 5 patches per image
                break
        
        if patches:
            all_patches[img_path] = patches
            
        # Memory management
        if idx % 50 == 0:
            import gc
            gc.collect()

    # Generate pairs - FIXED LOGIC
    pair_count = 0
    
    # 1. SAME-IMAGE PAIRS (AUTHENTIC) - Label 0
    for img_path in authentic_paths:
        if img_path not in all_patches or len(all_patches[img_path]) < 2:
            continue
            
        patches = all_patches[img_path]
        # Create pairs from same authentic image = AUTHENTIC (0)
        for i in range(min(2, len(patches))):
            for j in range(i+1, min(i+2, len(patches))):
                if pair_count >= max_pairs // 2:  # Reserve half for authentic
                    break
                pairs.append((patches[i], patches[j]))
                labels.append(0)  # Same image = authentic
                pair_count += 1
        if pair_count >= max_pairs // 2:
            break

    # 2. CROSS-IMAGE PAIRS (TAMPERED) - Label 1  
    # Use tampered images and pair with different images
    for tampered_path in tampered_paths:
        if tampered_path not in all_patches:
            continue
            
        tampered_patches = all_patches[tampered_path]
        
        # Find a different image to pair with
        for other_path in authentic_paths + tampered_paths:
            if other_path == tampered_path or other_path not in all_patches:
                continue
                
            other_patches = all_patches[other_path]
            
            # Create cross-image pairs = TAMPERED (1)
            for tp in tampered_patches[:2]:
                for op in other_patches[:2]:
                    if pair_count >= max_pairs:
                        break
                    pairs.append((tp, op))
                    labels.append(1)  # Different images = tampered
                    pair_count += 1
                    
            if pair_count >= max_pairs:
                break
        if pair_count >= max_pairs:
            break

    print(f"✅ Generated {len(pairs)} patch pairs ({sum(labels)} tampered, {len(labels)-sum(labels)} authentic)")
    
    # If we didn't get enough pairs, fill with same-image pairs
    if len(pairs) < max_pairs:
        print(f"⚠️ Only generated {len(pairs)} pairs. Filling with same-image pairs...")
        for img_path in split:
            if img_path not in all_patches or len(all_patches[img_path]) < 2:
                continue
                
            patches = all_patches[img_path]
            for i in range(min(1, len(patches))):
                for j in range(i+1, min(i+2, len(patches))):
                    if len(pairs) >= max_pairs:
                        break
                    pairs.append((patches[i], patches[j]))
                    label = 1 if img_path in gt_mapping else 0
                    labels.append(label)
            if len(pairs) >= max_pairs:
                break

    return pairs, labels

# ==============================
# 🔹 DEBUG SIFT EXTRACTION
# ==============================
def debug_sift_extraction(img_path):
    """Debug SIFT feature extraction for a single image"""
    print(f"\n🔍 Debugging: {os.path.basename(img_path)}")
    
    img = cv2.imread(img_path)
    if img is None:
        print("❌ Failed to load image")
        return
    
    print(f"Image shape: {img.shape}")
    
    kps, descs, gray, ent_norm, _ = extract_sift_on_entropy(img)
    
    if kps is None or len(kps) == 0:
        print("❌ No keypoints detected")
        print(f"Gray shape: {gray.shape}, range: [{gray.min()}, {gray.max()}]")
        print(f"Entropy shape: {ent_norm.shape}, range: [{ent_norm.min()}, {ent_norm.max()}]")
    else:
        print(f"✅ Detected {len(kps)} keypoints")
        print(f"Descriptor shape: {descs.shape}")
        
        # Test patch extraction
        successful_patches = 0
        for i, kp in enumerate(kps[:3]):  # Test first 3 keypoints
            patch = extract_patch(gray, kp, PATCH_SIZE)
            if patch is not None:
                successful_patches += 1
                print(f"  Keypoint {i}: patch shape {patch.shape}")
            else:
                print(f"  Keypoint {i}: failed to extract patch")
        
        print(f"Successful patches: {successful_patches}/{min(3, len(kps))}")

# ==============================
# 🔹 DATASET CLASS
# ==============================
class PairDataset(Dataset):
    def __init__(self, pairs, labels):
        self.pairs = pairs
        self.labels = labels

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        a, b = self.pairs[idx]
        a = torch.tensor(a, dtype=torch.float32).unsqueeze(0) / 255.0
        b = torch.tensor(b, dtype=torch.float32).unsqueeze(0) / 255.0
        #y = torch.tensor(self.labels[idx], dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)  # shape: scalar
        return a, b, y


# ==============================
# 🔹 SAVE / LOAD UTILITIES
# ==============================
def save_pairs(pairs, labels, path):
    joblib.dump({"pairs": pairs, "labels": labels}, path)
    print(f"💾 Saved {len(pairs)} pairs → {path}")


def load_pairs(path):
    data = joblib.load(path)
    print(f"📂 Loaded {len(data['pairs'])} pairs from {path}")
    return data["pairs"], data["labels"]

# ============================================================
# ✅ WRAPPER FUNCTION for backward compatibility
# ============================================================
def generate_pairs_from_split(split, gt_mapping, patch_size=PATCH_SIZE, max_pairs=5000):
    """
    Backward-compatible wrapper to generate patch pairs from the given split.
    Internally uses the main pair generation logic defined above.
    """
    print(f"[Wrapper] generate_pairs_from_split() called with {len(split)} images.")
    pairs, labels = generate_pairs(split, gt_mapping, patch_size, max_pairs)
    return pairs, labels


# ============================================================
# ✅ Main renamed generator
# ============================================================
def generate_pairs(split, gt_mapping, patch_size=PATCH_SIZE, max_pairs=5000):
    """
    Generates meaningful patch pairs for training (authentic vs tampered).
    """
    pairs, labels = [], []

    authentic_paths = [p for p in split if p not in gt_mapping]
    tampered_paths = [p for p in split if p in gt_mapping]

    print(f"🔍 Split: {len(authentic_paths)} authentic, {len(tampered_paths)} tampered")

    all_patches = {}
    for img_path in tqdm(split, desc="🧩 Extracting patches"):
        if len(pairs) >= max_pairs:
            break
        img = cv2.imread(img_path)
        if img is None:
            continue
        kps, descs, gray, _, _ = extract_sift_on_entropy(img)
        if not kps:
            continue
        patches = []
        for kp in kps[:5]:
            patch = extract_patch(gray, kp, patch_size)
            if patch is not None:
                patches.append(patch)
        if patches:
            all_patches[img_path] = patches

    # AUTHENTIC (same image pairs)
    for img_path in authentic_paths:
        if img_path not in all_patches:
            continue
        patches = all_patches[img_path]
        for i in range(min(2, len(patches))):
            for j in range(i + 1, min(i + 3, len(patches))):
                if len(pairs) >= max_pairs // 2:
                    break
                pairs.append((patches[i], patches[j]))
                labels.append(0)
        if len(pairs) >= max_pairs // 2:
            break

    # TAMPERED (cross image pairs)
    for t_path in tampered_paths:
        if t_path not in all_patches:
            continue
        t_patches = all_patches[t_path]
        for a_path in authentic_paths:
            if a_path not in all_patches:
                continue
            a_patches = all_patches[a_path]
            for tp in t_patches[:2]:
                for ap in a_patches[:2]:
                    if len(pairs) >= max_pairs:
                        break
                    pairs.append((tp, ap))
                    labels.append(1)
            if len(pairs) >= max_pairs:
                break
        if len(pairs) >= max_pairs:
            break

    print(f"✅ Generated {len(pairs)} pairs — {sum(labels)} tampered, {len(labels)-sum(labels)} authentic")
    return pairs, labels

