"""
Training script for EfficientNet deepfake detection model
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import sys
import pandas as pd
from tqdm import tqdm
import json
from efficientnet_pytorch import EfficientNet
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


class DeepfakeDataset(Dataset):
    """
    Dataset class for deepfake images
    """
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_data_loaders(data_dir, batch_size=32, img_size=224, test_size=0.2, val_size=0.1):
    """
    Create data loaders for training, validation, and testing
    
    Args:
        data_dir: Root directory containing 'real' and 'fake' subdirectories
        batch_size: Batch size for training
        img_size: Target image size
        test_size: Proportion of data for testing
        val_size: Proportion of training data for validation
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """

    train_transform = transforms.Compose([
        transforms.Resize(256),  # Resize shorter side to 256, maintaining aspect ratio
        transforms.CenterCrop(img_size),  # Crop center 224x224
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize(256),  # Resize shorter side to 256, maintaining aspect ratio
        transforms.CenterCrop(img_size),  # Crop center 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Collect image paths and labels
    image_paths = []
    labels = []
    
    real_dir = os.path.join(data_dir, 'real')
    fake_dir = os.path.join(data_dir, 'fake')
    
    # Load real images
    if os.path.exists(real_dir):
        for img_file in os.listdir(real_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(real_dir, img_file))
                labels.append(0)  # 0 for real
    
    # Load fake images
    if os.path.exists(fake_dir):
        for img_file in os.listdir(fake_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(fake_dir, img_file))
                labels.append(1)  # 1 for fake
    
    if len(image_paths) == 0:
        raise ValueError(f"No images found in {data_dir}. Please check the directory structure.")
    
    print(f"Found {len(image_paths)} images: {labels.count(0)} real, {labels.count(1)} fake")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        image_paths, labels, test_size=test_size, random_state=42, stratify=labels
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=42, stratify=y_train
    )
    
    # Create datasets
    train_dataset = DeepfakeDataset(X_train, y_train, transform=train_transform)
    val_dataset = DeepfakeDataset(X_val, y_val, transform=val_test_transform)
    test_dataset = DeepfakeDataset(X_test, y_test, transform=val_test_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader


def train_model(model, train_loader, val_loader, num_epochs=10, device='cpu', 
                learning_rate=0.001, save_path='saved_models', unfreeze_epoch=7):
    """
    Train the model with progressive unfreezing
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        device: Device to train on
        learning_rate: Learning rate for optimizer
        save_path: Directory to save model checkpoints
        unfreeze_epoch: Epoch at which to unfreeze last blocks (default: 7)
    """
    os.makedirs(save_path, exist_ok=True)
    
    criterion = nn.CrossEntropyLoss()
    
    # Create optimizer with different learning rates for classifier and backbone
    # Initially only classifier is trainable
    classifier_params = list(model.backbone._fc.parameters())
    optimizer = optim.Adam(classifier_params, lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Progressive unfreezing: unfreeze last blocks at specified epoch
        if epoch == unfreeze_epoch:
            unfreeze_last_blocks(model, num_blocks=2)
            
            # Recreate optimizer with different learning rates
            # Lower LR for unfrozen backbone, higher for classifier
            backbone_params = []
            for block in model.backbone._blocks[-2:]:  # Last 2 blocks
                backbone_params.extend(list(block.parameters()))
            
            classifier_params = list(model.backbone._fc.parameters())
            
            # Use differential learning rates: higher for classifier, lower for backbone
            optimizer = optim.Adam([
                {'params': classifier_params, 'lr': learning_rate},
                {'params': backbone_params, 'lr': learning_rate * 0.1}  # 10x lower for backbone
            ])
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
            print(f"  Using differential learning rates: {learning_rate} (classifier), {learning_rate * 0.1} (backbone)")
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            train_pbar.set_postfix({'loss': loss.item(), 'acc': 100*train_correct/train_total})
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_pbar.set_postfix({'loss': loss.item(), 'acc': 100*val_correct/val_total})
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        val_losses.append(avg_val_loss)
        
        scheduler.step(avg_val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_acc': val_acc,
                'train_loss': avg_train_loss,
                'train_acc': train_acc
            }
            torch.save(checkpoint, os.path.join(save_path, 'best_model.pth'))
            print(f'  Saved best model (Val Loss: {avg_val_loss:.4f})')
    
    return train_losses, val_losses


def freeze_backbone_layers(model):
    """
    Freeze all backbone layers except the classifier head
    
    Args:
        model: DeepfakeDetector model
    """
    # Freeze all parameters in the backbone
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    # Unfreeze the classifier head (final linear layer)
    for param in model.backbone._fc.parameters():
        param.requires_grad = True
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Frozen backbone layers. Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")


def unfreeze_last_blocks(model, num_blocks=2):
    """
    Unfreeze the last N blocks of EfficientNet backbone
    
    Args:
        model: DeepfakeDetector model
        num_blocks: Number of last blocks to unfreeze (default: 2)
    """
    # Get total number of blocks
    total_blocks = len(model.backbone._blocks)
    
    # Unfreeze the last num_blocks
    blocks_to_unfreeze = total_blocks - num_blocks
    
    print(f"\nUnfreezing last {num_blocks} blocks (blocks {blocks_to_unfreeze} to {total_blocks-1})...")
    
    # Unfreeze specified blocks
    for i in range(blocks_to_unfreeze, total_blocks):
        for param in model.backbone._blocks[i].parameters():
            param.requires_grad = True
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Unfrozen last {num_blocks} blocks. Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")


def main():
    """
    Main training function
    """
    # Configuration
    DATA_DIR = 'data'  # Should contain 'real' and 'fake' subdirectories
    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    IMG_SIZE = 224
    LEARNING_RATE = 0.001
    MODEL_NAME = 'efficientnet-b0'
    SAVE_PATH = 'saved_models'
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader = get_data_loaders(
        DATA_DIR, batch_size=BATCH_SIZE, img_size=IMG_SIZE
    )
    
    # Create model
    print(f"Creating {MODEL_NAME} model...")
    # Import here after path is set
    from src.models.model_loader import DeepfakeDetector
    model = DeepfakeDetector(model_name=MODEL_NAME, num_classes=2, pretrained=True)
    
    # Freeze backbone layers (only train classifier head)
    print("Freezing backbone layers...")
    freeze_backbone_layers(model)
    
    model.to(device)
    
    # Train model
    print("Starting training...")
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, 
        num_epochs=NUM_EPOCHS, device=device,
        learning_rate=LEARNING_RATE, save_path=SAVE_PATH
    )
    
    print("Training completed!")
    print(f"Best model saved to {SAVE_PATH}/best_model.pth")


if __name__ == '__main__':
    main()

