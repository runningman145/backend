"""
Model loading and caching.
Handles YOLO and ReID model initialization and caching.
"""
import torch
from flask import current_app
from ultralytics import YOLO
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

# Global model instances (loaded once at startup)
_models_cache = {
    'yolo': None,
    'reid': None,
    'transform': None,
    'device': None,
}

# Model configuration
MODEL_CONFIG = {
    'FRAME_SKIP': 15,
    'MIN_BOX_AREA': 5000,
    'MATCH_THRESHOLD_PERCENT': 40,
    'YOLO_IMGSZ': 640,
    'REID_INPUT_SIZE': (256, 128),
    'VEHICLE_CLASSES': [2, 5, 7],  # car, bus, truck
}


def get_device():
    """Get the appropriate device (GPU or CPU)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():  # For Apple Silicon Macs
        return torch.device('mps')
    else:
        return torch.device('cpu')


def load_models():
    """Load YOLO and ReID models (called once at startup)."""
    if _models_cache['yolo'] is not None:
        return  # Already loaded

    device = get_device()
    _models_cache['device'] = device
    
    # Check if we're in Flask context for logging
    try:
        current_app.logger.info(f"Loading models on device: {device}")
    except RuntimeError:
        # Not in Flask context, use print instead
        print(f"Loading models on device: {device}")
    
    # Load YOLO model
    try:
        _models_cache['yolo'] = YOLO("yolov8n.pt")
        if hasattr(current_app, 'logger'):
            current_app.logger.info("YOLO model loaded successfully")
    except Exception as e:
        if hasattr(current_app, 'logger'):
            current_app.logger.error(f"Failed to load YOLO model: {str(e)}")
        else:
            print(f"Failed to load YOLO model: {str(e)}")
        raise
    
    # Load ReID model
    try:
        # Use pretrained weights for better feature extraction
        reid_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        # Remove the final classification layer to get embeddings
        reid_model.fc = torch.nn.Identity()
        
        # Try to load custom trained weights if provided
        reid_model_path = None
        try:
            reid_model_path = current_app.config.get('REID_MODEL_PATH')
        except RuntimeError:
            # Not in Flask context
            pass
            
        if reid_model_path:
            try:
                # Load state dict with proper device mapping
                state_dict = torch.load(reid_model_path, map_location=device)
                reid_model.load_state_dict(state_dict)
                if hasattr(current_app, 'logger'):
                    current_app.logger.info(f"ReID model loaded from {reid_model_path}")
            except FileNotFoundError:
                if hasattr(current_app, 'logger'):
                    current_app.logger.warning(f"ReID model not found at {reid_model_path}, using pretrained ResNet50")
                else:
                    print(f"ReID model not found at {reid_model_path}, using pretrained ResNet50")
            except Exception as e:
                if hasattr(current_app, 'logger'):
                    current_app.logger.warning(f"Failed to load custom ReID model: {str(e)}, using pretrained ResNet50")
                else:
                    print(f"Failed to load custom ReID model: {str(e)}, using pretrained ResNet50")
        else:
            if hasattr(current_app, 'logger'):
                current_app.logger.info("No custom ReID model path configured, using pretrained ResNet50")
        
        reid_model = reid_model.to(device)
        reid_model.eval()
        _models_cache['reid'] = reid_model
        
    except Exception as e:
        if hasattr(current_app, 'logger'):
            current_app.logger.error(f"Failed to load ReID model: {str(e)}")
        else:
            print(f"Failed to load ReID model: {str(e)}")
        raise
    
    # Set up transforms for ReID model
    _models_cache['transform'] = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(MODEL_CONFIG['REID_INPUT_SIZE']),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    try:
        if hasattr(current_app, 'logger'):
            current_app.logger.info("Models loaded successfully")
        else:
            print("Models loaded successfully")
    except RuntimeError:
        print("Models loaded successfully")


def get_models():
    """Get cached models, loading if necessary."""
    load_models()
    return _models_cache


def unload_models():
    """Unload models to free memory (useful for testing or hot reload)."""
    global _models_cache
    _models_cache = {
        'yolo': None,
        'reid': None,
        'transform': None,
        'device': None,
    }
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        if hasattr(current_app, 'logger'):
            current_app.logger.info("Models unloaded successfully")
    except (RuntimeError, AttributeError):
        pass


def get_model_info():
    """Get information about loaded models."""
    info = {
        'device': str(_models_cache['device']) if _models_cache['device'] else 'Not loaded',
        'yolo_loaded': _models_cache['yolo'] is not None,
        'reid_loaded': _models_cache['reid'] is not None,
        'transform_loaded': _models_cache['transform'] is not None,
        'config': MODEL_CONFIG.copy()
    }
    
    # Add CUDA info if available
    if torch.cuda.is_available():
        info['cuda_available'] = True
        info['cuda_device_count'] = torch.cuda.device_count()
        info['cuda_device_name'] = torch.cuda.get_device_name(0)
    else:
        info['cuda_available'] = False
    
    return info


# Optional: Warm up models to ensure they're ready
def warmup_models():
    """Run a dummy inference to warm up the models."""
    try:
        models = get_models()
        device = models['device']
        
        # Warm up YOLO with dummy input
        if models['yolo']:
            dummy_img = torch.rand(1, 3, MODEL_CONFIG['YOLO_IMGSZ'], MODEL_CONFIG['YOLO_IMGSZ'])
            # YOLO expects numpy array or path, so we skip warmup for now
        
        # Warm up ReID model
        if models['reid'] and models['transform']:
            dummy_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            dummy_tensor = models['transform'](dummy_img).unsqueeze(0).to(device)
            with torch.no_grad():
                _ = models['reid'](dummy_tensor)
        
        try:
            if hasattr(current_app, 'logger'):
                current_app.logger.info("Models warmed up successfully")
        except (RuntimeError, AttributeError):
            pass
            
    except Exception as e:
        try:
            if hasattr(current_app, 'logger'):
                current_app.logger.warning(f"Model warmup failed: {str(e)}")
        except (RuntimeError, AttributeError):
            pass


# Import numpy for warmup function
import numpy as np