"""
Image processing utilities for deepfake detection
"""
import cv2
import numpy as np
import os
from PIL import Image
import torch
from torchvision import transforms


def preprocess_image(image_path, img_size=224):
    """
    Preprocess image for EfficientNet model
    
    Args:
        image_path: Path to the image file
        img_size: Target image size (default 224 for EfficientNet)
    
    Returns:
        Preprocessed tensor ready for model input
    """
    # Read image
    image = Image.open(image_path).convert('RGB')
    
    # Define transformations for EfficientNet
    # Resize to 256 (maintains aspect ratio), then center crop to 224x224
    transform = transforms.Compose([
        transforms.Resize(256),  # Resize shorter side to 256, maintaining aspect ratio
        transforms.CenterCrop(img_size),  # Crop center 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Apply transformations
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    return image_tensor


def detect_faces(image_path):
    """
    Detect faces in an image using OpenCV's Haar Cascade
    
    Args:
        image_path: Path to the image file
    
    Returns:
        tuple: (has_face, num_faces, error_message)
        has_face: True if at least one face is detected
        num_faces: Number of faces detected
        error_message: Error message if detection fails
    """
    try:
        # Read image with OpenCV
        img = cv2.imread(image_path)
        if img is None:
            return False, 0, "Could not read image file"
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Load the face cascade classifier
        # Try to find the cascade file in common locations
        cascade_paths = [
            '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
            '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
            'haarcascade_frontalface_default.xml'
        ]
        
        face_cascade = None
        for path in cascade_paths:
            if os.path.exists(path):
                face_cascade = cv2.CascadeClassifier(path)
                break
        
        if face_cascade is None:
            # Try using cv2's built-in path
            try:
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            except:
                return False, 0, "Face detection classifier not available"
        
        if face_cascade.empty():
            return False, 0, "Face detection classifier failed to load"
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        num_faces = len(faces)
        return num_faces > 0, num_faces, None
        
    except Exception as e:
        return False, 0, f"Face detection error: {str(e)}"


def validate_image(image_path):
    """
    Validate that the uploaded file is a valid image
    
    Args:
        image_path: Path to the image file
    
    Returns:
        tuple: (is_valid, error_message)
    """
    try:
        image = Image.open(image_path)
        image.verify()
        return True, None
    except Exception as e:
        return False, f"Invalid image file: {str(e)}"


