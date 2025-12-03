"""
Model loading and prediction utilities
"""
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import os


class DeepfakeDetector(nn.Module):
    """
    EfficientNet-based deepfake detector
    """
    def __init__(self, model_name='efficientnet-b0', num_classes=2, pretrained=True):
        super(DeepfakeDetector, self).__init__()
        # Load EfficientNet backbone
        self.backbone = EfficientNet.from_pretrained(model_name) if pretrained else EfficientNet.from_name(model_name)
        
        # Replace the classifier for binary classification
        num_features = self.backbone._fc.in_features
        self.backbone._fc = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)


def load_model(model_path, device='cpu', model_name='efficientnet-b0'):
    """
    Load a trained model from disk
    
    Args:
        model_path: Path to the saved model file
        device: Device to load model on ('cpu' or 'cuda')
        model_name: EfficientNet variant name
    
    Returns:
        Loaded model in evaluation mode
    """
    model = DeepfakeDetector(model_name=model_name, num_classes=2, pretrained=False)
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Model loaded from {model_path}")
    else:
        print(f"Warning: Model file not found at {model_path}. Using untrained model.")
    
    model.eval()
    model.to(device)
    return model


def predict_image(model, image_tensor, device='cpu'):
    """
    Make prediction on a single image
    
    Args:
        model: Trained model
        image_tensor: Preprocessed image tensor
        device: Device to run inference on
    
    Returns:
        tuple: (prediction, confidence_score)
        prediction: 0 for real, 1 for fake
        confidence_score: Confidence percentage (0-100)
    """
    model.eval()
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
        # Get confidence as percentage
        confidence_score = confidence.item() * 100
        
        # Get probabilities for both classes
        real_prob = probabilities[0][0].item() * 100
        fake_prob = probabilities[0][1].item() * 100
        
        prediction = predicted.item()
        label = "Fake" if prediction == 1 else "Real"
        
    return {
        'prediction': label,
        'is_fake': prediction == 1,
        'confidence': round(confidence_score, 2),
        'real_probability': round(real_prob, 2),
        'fake_probability': round(fake_prob, 2)
    }


