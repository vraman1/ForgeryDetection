# inference_gradcam_fixed.py - FIXED VERSION
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import warnings
import os
import glob
warnings.filterwarnings('ignore')

class BinaryClassifierWithGradCAM(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        self.backbone.classifier[1] = nn.Linear(1280, num_classes)
        
        # Grad-CAM hooks
        self.gradients = None
        self.activations = None
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
            
        def forward_hook(module, input, output):
            self.activations = output
            
        self.backbone.features[-1].register_forward_hook(forward_hook)
        self.backbone.features[-1].register_full_backward_hook(backward_hook)
    
    def forward(self, x):
        return self.backbone(x)

def generate_gradcam(model, image_tensor, target_class=None):
    """Generate Grad-CAM heatmap - SIMPLIFIED"""
    model.eval()
    
    # Forward pass
    output = model(image_tensor)
    
    if target_class is None:
        target_class = output.argmax(dim=1).item()
    
    # Backward pass
    model.zero_grad()
    one_hot = torch.zeros_like(output)
    one_hot[0][target_class] = output[0][target_class]  # Use actual score
    output.backward(gradient=one_hot)
    
    # Get gradients and activations
    gradients = model.gradients.detach().cpu().numpy()[0]  # [1280, 7, 7]
    activations = model.activations.detach().cpu().numpy()[0]  # [1280, 7, 7]
    
    # Global average pooling of gradients
    weights = np.mean(gradients, axis=(1, 2))
    
    # Weighted combination of activation maps
    cam = np.zeros(activations.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * activations[i]
    
    # Apply ReLU and resize
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (image_tensor.shape[3], image_tensor.shape[2]))
    cam = (cam - np.min(cam)) / (np.max(cam) + 1e-8)
    
    return cam, output

def create_localized_heatmap_fixed(original_image, prediction, confidence, gradcam_heatmap=None):
    """FIXED VERSION - Pale pink circles with shading"""
    if isinstance(original_image, Image.Image):
        img_np = np.array(original_image)
    else:
        img_np = original_image.copy()
    
    # Ensure proper shape and type
    if len(img_np.shape) == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    elif img_np.shape[2] == 4:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
    
    height, width = img_np.shape[:2]
    
    if prediction == 1 and gradcam_heatmap is not None:  
        # TAMPERED - Use Grad-CAM for localization with pale pink circles
        
        # Resize Grad-CAM
        cam_resized = cv2.resize(gradcam_heatmap, (width, height))
        
        # Create pale pink color (BGR format - light pink)
        # pale_pink = [203, 192, 255]  # BGR: [203, 192, 255] = RGB: [255, 192, 203]
        pale_pink = [187, 111, 110]  # BGR: [203, 192, 255] = RGB: [255, 192, 203]
        
        # Create output image
        blended = img_np.copy()
        
        # Find high-activation regions and draw pale pink circles
        activation_threshold = 0.3
        high_activation_mask = cam_resized > activation_threshold
        
        if np.any(high_activation_mask):
            # Find contours of high activation regions
            activation_binary = (high_activation_mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(activation_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if cv2.contourArea(contour) > 50:  # Filter small regions
                    # Get bounding circle
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    center = (int(x), int(y))
                    radius = int(radius * 1.2)  # Slightly larger circle
                    
                    if radius > 10:  # Only draw meaningful circles
                        # Create circle mask
                        circle_mask = np.zeros((height, width), dtype=np.uint8)
                        cv2.circle(circle_mask, center, radius, 255, -1)
                        
                        # Create pale pink overlay
                        pink_overlay = np.zeros_like(blended)
                        pink_overlay[:] = pale_pink
                        
                        # Apply gradient shading (darker at edges, lighter in center)
                        gradient_mask = np.zeros((height, width), dtype=np.float32)
                        cv2.circle(gradient_mask, center, radius, 1.0, -1)
                        
                        # Create distance transform for gradient
                        dist_mask = np.zeros((height, width), dtype=np.uint8)
                        cv2.circle(dist_mask, center, radius, 255, -1)
                        distance = cv2.distanceTransform(dist_mask, cv2.DIST_L2, 5)
                        distance = np.clip(distance / radius, 0, 1)  # Normalize
                        
                        # Invert so center is brighter
                        gradient_alpha = 1 - distance
                        gradient_alpha = gradient_alpha * 0.6  # Overall transparency
                        
                        # Apply the shaded pink overlay
                        circle_area = gradient_mask > 0
                        for c in range(3):
                            blended[circle_area, c] = (
                                blended[circle_area, c] * (1 - gradient_alpha[circle_area]) + 
                                pink_overlay[circle_area, c] * gradient_alpha[circle_area]
                            )
        
        label = "TAMPERED"
        text_color = (255, 255, 255)
        bg_color = (0, 0, 200)  # Keep blue background for text
        
    elif prediction == 1:
        # Fallback: Tampered but no Grad-CAM
        blended = img_np.copy()
        label = "TAMPERED"
        text_color = (255, 255, 255)
        bg_color = (0, 0, 200)
        
    else:
        # AUTHENTIC - Keep original image with minimal overlay
        blended = img_np.copy()
        label = "AUTHENTIC"
        text_color = (255, 255, 255)
        bg_color = (200, 0, 0)  # Red background for authentic
    
    # Add text (unchanged)
    text = f"{label} ({confidence:.2%})"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = min(1.2, width / 400)
    thickness = max(2, int(font_scale * 2))
    
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    
    padding = 10
    text_bg = np.full((text_height + padding*2, text_width + padding*2, 3), bg_color, dtype=np.uint8)
    
    if text_height + padding*2 <= height and text_width + padding*2 <= width:
        text_region = blended[padding:padding+text_bg.shape[0], padding:padding+text_bg.shape[1]]
        blended[padding:padding+text_bg.shape[0], padding:padding+text_bg.shape[1]] = cv2.addWeighted(
            text_region, 0.3, text_bg, 0.7, 0
        )
        
        text_x = padding + 5
        text_y = padding + text_height + 5
        cv2.putText(blended, text, (text_x, text_y), font, font_scale, text_color, thickness)
    
    return blended

def load_model(model_path, device):
    """Load model with Grad-CAM support"""
    try:
        model = BinaryClassifierWithGradCAM(num_classes=2)
        checkpoint = torch.load(model_path, map_location=device)
        
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
                
        model.load_state_dict(new_state_dict, strict=False)
        model.to(device)
        model.eval()
        print("✅ Model loaded with Grad-CAM support!")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def predict_image_with_localization(model, image_path, device, image_size=224):
    """Predict with localization - FIXED"""
    try:
        original_image = Image.open(image_path).convert('RGB')
        
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(original_image).unsqueeze(0).to(device)
        image_tensor.requires_grad = True
        
        # Get Grad-CAM
        gradcam_heatmap, outputs = generate_gradcam(model, image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
        
        # Create localized heatmap
        heatmap_image = create_localized_heatmap_fixed(original_image, predicted_class, confidence, gradcam_heatmap)
        
        return predicted_class, confidence, heatmap_image, original_image
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        raise

def save_single_result(heatmap, image_path, output_dir="detection_results"):
    """Save result with proper filename"""
    os.makedirs(output_dir, exist_ok=True)
    original_filename = os.path.basename(image_path)
    name, ext = os.path.splitext(original_filename)
    new_filename = f"{name}_result{ext}"
    output_path = os.path.join(output_dir, new_filename)
    
    heatmap_bgr = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, heatmap_bgr)
    return output_path

def display_menu():
    """Display the main menu"""
    print("\n" + "="*50)
    print("🎯 IMAGE TAMPER DETECTION SYSTEM")
    print("="*50)
    print("1. Test single image")
    print("2. Test multiple images in folder")
    print("3. Change model path (current: saved_models/best_hybrid_classifier.pth)")
    print("4. Exit")
    print("="*50)

def test_multiple_images(model, device, folder_path):
    """Test all images in a folder"""
    if not os.path.exists(folder_path):
        print("❌ Folder not found!")
        return
    
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []
    
    for extension in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, extension)))
        image_files.extend(glob.glob(os.path.join(folder_path, extension.upper())))
    
    if not image_files:
        print("❌ No images found in folder!")
        return
    
    print(f"📁 Found {len(image_files)} images")
    
    for i, image_path in enumerate(image_files, 1):
        try:
            print(f"\n🔍 Processing {i}/{len(image_files)}: {os.path.basename(image_path)}")
            pred_class, confidence, heatmap, original = predict_image_with_localization(model, image_path, device)
            
            status = "TAMPERED" if pred_class == 1 else "AUTHENTIC"
            print(f"   Result: {status} (Confidence: {confidence:.2%})")
            
            # Save result
            output_path = save_single_result(heatmap, image_path)
            print(f"   💾 Saved: {os.path.basename(output_path)}")
            
        except Exception as e:
            print(f"   ❌ Error processing {image_path}: {e}")
    
    print(f"\n✅ Completed processing {len(image_files)} images!")

# ============================================================
# 🎯 MAIN APPLICATION WITH MENU
# ============================================================
if __name__ == "__main__":
    MODEL_PATH = "saved_models/best_hybrid_classifier.pth"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"🖥️ Using device: {DEVICE}")
    print("🎨 Using PALE PINK CIRCLES with shading for tampered regions")
    
    model = load_model(MODEL_PATH, DEVICE)
    if model is None:
        print("❌ Cannot continue without model")
        exit(1)
    
    while True:
        display_menu()
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            # Test single image
            image_path = input("Enter image path to test: ").strip()
            if os.path.exists(image_path):
                try:
                    pred_class, confidence, heatmap, original = predict_image_with_localization(model, image_path, DEVICE)
                    
                    # Display
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                    ax1.imshow(original)
                    ax1.set_title('Original Image', fontweight='bold')
                    ax1.axis('off')
                    
                    ax2.imshow(heatmap)
                    status = "TAMPERED" if pred_class == 1 else "AUTHENTIC"
                    ax2.set_title(f'{status} - Confidence: {confidence:.2%}', fontweight='bold')
                    ax2.axis('off')
                    
                    plt.tight_layout()
                    plt.show()
                    
                    # Save
                    output_path = save_single_result(heatmap, image_path)
                    print(f"💾 Saved: {output_path}")
                    
                    if pred_class == 1:
                        print("🎀 Pale pink circles show WHERE tampering was detected")
                        print("💫 Circles have gradient shading (lighter in center)")
                    else:
                        print("🔵 Image marked as authentic - no tampering detected")
                    
                except Exception as e:
                    print(f"❌ Error: {e}")
            else:
                print("❌ Image not found!")
                
        elif choice == '2':
            # Test multiple images
            folder_path = input("Enter folder path containing images: ").strip()
            test_multiple_images(model, DEVICE, folder_path)
            
        elif choice == '3':
            # Change model path
            new_model_path = input("Enter new model path: ").strip()
            if os.path.exists(new_model_path):
                MODEL_PATH = new_model_path
                model = load_model(MODEL_PATH, DEVICE)
                if model is None:
                    print("❌ Failed to load new model, reverting to previous")
                    MODEL_PATH = "saved_models/best_hybrid_classifier.pth"
                    model = load_model(MODEL_PATH, DEVICE)
            else:
                print("❌ Model file not found!")
                
        elif choice == '4':
            print("👋 Thank you for using Image Tamper Detection System!")
            break
            
        else:
            print("❌ Invalid choice! Please enter 1-4")
        
        input("\nPress Enter to continue...")