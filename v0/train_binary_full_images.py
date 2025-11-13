# train_hybrid_balanced.py - HYBRID APPROACH (2:1 Ratio)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import os
from PIL import Image
import warnings
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import random

# ============================================================
# 🔇 SUPPRESS WARNINGS
# ============================================================
warnings.filterwarnings('ignore')
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ============================================================
# ⚙️ CONFIGURATION
# ============================================================
DATA_PATH = "data/CASIA2"
BATCH_SIZE = 16
IMAGE_SIZE = 224
NUM_EPOCHS = 12
LEARNING_RATE = 1e-4
HYBRID_RATIO = 2  # 2:1 ratio (Authentic:Tampered)

# ============================================================
# 📦 HYBRID DATASET CLASS
# ============================================================
class HybridCASIADataset(Dataset):
    def __init__(self, image_paths, gt_mapping, transform=None):
        self.image_paths = []
        self.labels = []
        
        for path in image_paths:
            if os.path.exists(path):
                self.image_paths.append(path)
                label = 1 if path in gt_mapping else 0
                self.labels.append(label)
        
        self.transform = transform
        
        # Print hybrid distribution
        class_counts = np.bincount(self.labels)
        print(f"📊 Hybrid Dataset: {len(self.image_paths)} images")
        print(f"   Authentic (0): {class_counts[0]} images")
        print(f"   Tampered (1): {class_counts[1]} images")
        print(f"   Achieved Ratio: {class_counts[0]/class_counts[1]:.1f}:1")
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB')
            label = self.labels[idx]
            
            if self.transform:
                image = self.transform(image)
                
            return image, label
        except Exception as e:
            # Return proper image instead of zeros
            dummy_image = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), color='gray')
            if self.transform:
                dummy_image = self.transform(dummy_image)
            return dummy_image, 0

# ============================================================
# 🧠 BINARY CLASSIFIER
# ============================================================
class BinaryClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights='DEFAULT')
        self.backbone.classifier[1] = nn.Linear(1280, num_classes)
    
    def forward(self, x):
        return self.backbone(x)

# ============================================================
# 📊 HYBRID DATA PREPARATION (2:1 Ratio)
# ============================================================
def prepare_hybrid_data_loaders():
    from data_utils import load_casia_dataset, split_dataset
    
    authentic_paths, tampered_paths, gt_mapping = load_casia_dataset(DATA_PATH)
    
    print(f"📁 Original dataset:")
    print(f"   Authentic: {len(authentic_paths)}, Tampered: {len(tampered_paths)}")
    print(f"   Natural ratio: {len(authentic_paths)/len(tampered_paths):.2f}:1")
    
    # HYBRID APPROACH: Create 2:1 ratio
    # Use all tampered images, take 2x authentic images
    hybrid_authentic_count = min(len(authentic_paths), len(tampered_paths) * HYBRID_RATIO)
    hybrid_authentic = authentic_paths[:hybrid_authentic_count]
    hybrid_tampered = tampered_paths  # Use all tampered
    
    print(f"\n⚖️  Creating Hybrid Dataset (2:1 ratio):")
    print(f"   Using {len(hybrid_tampered)} tampered images")
    print(f"   Using {len(hybrid_authentic)} authentic images")
    print(f"   Target ratio: {HYBRID_RATIO}:1")
    print(f"   Total hybrid dataset: {len(hybrid_authentic) + len(hybrid_tampered)} images")
    
    # Split the hybrid dataset
    dataset_split = split_dataset(hybrid_authentic, hybrid_tampered, test_size=0.3)
    
    print(f"\n📊 Final Hybrid Splits:")
    print(f"   Train: {len(dataset_split['train'])} images")
    print(f"   Test: {len(dataset_split['test'])} images")
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = HybridCASIADataset(dataset_split['train'], gt_mapping, transform=train_transform)
    test_dataset = HybridCASIADataset(dataset_split['test'], gt_mapping, transform=test_transform)
    
    # Simple dataloaders - no complex weighting needed!
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, test_loader

# ============================================================
# 📈 ENHANCED VALIDATION METRICS
# ============================================================
def validate_with_metrics(model, test_loader, device, epoch):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = 100 * (np.array(all_preds) == np.array(all_labels)).mean()
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, 
                                 target_names=['Authentic', 'Tampered'],
                                 digits=4, output_dict=True)
    
    # Detailed analysis
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) * 100
    fnr = fn / (fn + tp) * 100
    
    print(f"\n📊 Epoch {epoch} - Hybrid Performance:")
    print("=" * 50)
    print(f"Overall Accuracy: {accuracy:.2f}%")
    
    print(f"\nConfusion Matrix (2:1 ratio aware):")
    print("         Predicted")
    print("         Authentic  Tampered")
    print(f"Actual A [  {tn:5d}    {fp:5d}  ]")
    print(f"        T [  {fn:5d}    {tp:5d}  ]")
    
    print(f"\n🎯 Critical Metrics for Forgery Detection:")
    print(f"   • Tampered Recall: {report['Tampered']['recall']*100:.2f}%")
    print(f"   • Tampered Precision: {report['Tampered']['precision']*100:.2f}%")
    print(f"   • False Negative Rate: {fnr:.2f}%")
    print(f"   • False Positive Rate: {fpr:.2f}%")
    print(f"   • F1-Score: {report['Tampered']['f1-score']:.3f}")
    
    return accuracy, cm, report

# ============================================================
# 🏋️ HYBRID TRAINING FUNCTION
# ============================================================
def train_hybrid_classifier():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")
    
    # Prepare hybrid data (2:1 ratio)
    train_loader, test_loader = prepare_hybrid_data_loaders()
    
    # Model
    model = BinaryClassifier()
    model.to(device)
    
    # Freeze early layers for faster training
    for param in model.backbone.features[:-3].parameters():
        param.requires_grad = False
    
    # SIMPLE LOSS - no complex weighting needed!
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(trainable_params, lr=LEARNING_RATE, weight_decay=1e-5)
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [], 'val_acc': [], 
        'val_tampered_recall': [], 'best_epoch': 0, 
        'best_tampered_recall': 0, 'best_overall_accuracy': 0,
        'approach': 'hybrid_2_1'
    }
    
    # Create directories
    os.makedirs("saved_models/hybrid_models", exist_ok=True)
    
    print(f"\n🚀 Starting Hybrid Training (2:1 ratio) for {NUM_EPOCHS} epochs...")
    print("✅ No complex weighting - Simple balanced approach")
    print("=" * 60)
    
    for epoch in range(NUM_EPOCHS):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_accuracy = 100 * correct / total
        avg_loss = running_loss / len(train_loader)
        
        # Validation with metrics
        val_accuracy, cm, report = validate_with_metrics(model, test_loader, device, epoch+1)
        tampered_recall = report['Tampered']['recall']
        
        # Save history
        history['train_loss'].append(avg_loss)
        history['train_acc'].append(train_accuracy)
        history['val_acc'].append(val_accuracy)
        history['val_tampered_recall'].append(tampered_recall)
        
        print(f"\n📅 Epoch {epoch+1}/{NUM_EPOCHS} Summary:")
        print(f"   Train Loss: {avg_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
        print(f"   Val Acc: {val_accuracy:.2f}%, Tampered Recall: {tampered_recall*100:.2f}%")
        
        # Save all epoch models
        epoch_model_path = f"saved_models/hybrid_models/model_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), epoch_model_path)
        
        # Save best model based on tampered recall
        if tampered_recall > history['best_tampered_recall']:
            history['best_tampered_recall'] = tampered_recall
            history['best_epoch'] = epoch + 1
            history['best_overall_accuracy'] = val_accuracy
            torch.save(model.state_dict(), "saved_models/best_hybrid_classifier.pth")
            print(f"   🏆 NEW BEST! Tampered Recall: {tampered_recall*100:.2f}%")
        
        print("-" * 60)
        
        # Early stopping if performance plateaus
        if epoch >= 5 and tampered_recall > 0.75 and abs(history['val_acc'][-1] - history['val_acc'][-2]) < 0.5:
            print("🎯 Performance plateau reached - consider stopping")
    
    # Final summary
    print(f"\n🎉 Hybrid Training Complete!")
    print(f"   Best Epoch: {history['best_epoch']}")
    print(f"   Best Overall Accuracy: {history['best_overall_accuracy']:.2f}%")
    print(f"   Best Tampered Recall: {history['best_tampered_recall']*100:.2f}%")
    print(f"   Models saved in: saved_models/hybrid_models/")
    
    # Save training history
    joblib.dump(history, "saved_models/hybrid_training_history.pkl")
    
    return history

# ============================================================
# 🎯 MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    os.makedirs("saved_models/hybrid_models", exist_ok=True)
    
    print("🚀 Starting HYBRID Binary Classification (2:1 Ratio)")
    print("✅ Using 2x Authentic : 1x Tampered - No Complex Weighting")
    
    history = train_hybrid_classifier()