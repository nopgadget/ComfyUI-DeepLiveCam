import os
import cv2
import numpy as np
import torch
import sys
import logging
from typing import Any
import onnxruntime

# Add DeepLiveCam to path
dlc_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DeepLiveCam")
sys.path.append(dlc_path)

# Import DeepLiveCam modules
from .DeepLiveCam.modules.face_analyser import get_one_face, get_many_faces
from .DeepLiveCam.modules.processors.frame.face_swapper import get_face_swapper, swap_face
from .DeepLiveCam.modules.utilities import conditional_download
from .DeepLiveCam.modules import globals as dlc_globals

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('DeepLiveCamNode')

def get_available_execution_providers():
    """Get the list of available execution providers for ONNX Runtime."""
    providers = onnxruntime.get_available_providers()
    # Convert to more friendly names for UI display
    provider_map = {
        "CPUExecutionProvider": "CPU",
        "CUDAExecutionProvider": "CUDA (NVIDIA GPU)",
        "ROCMExecutionProvider": "ROCm (AMD GPU)",
        "CoreMLExecutionProvider": "CoreML (Apple)",
        "DmlExecutionProvider": "DirectML",
        "OpenVINOExecutionProvider": "OpenVINO"
    }
    
    available_providers = []
    for provider in providers:
        friendly_name = provider_map.get(provider, provider)
        available_providers.append((provider, friendly_name))
    
    return available_providers

class DeepLiveCamNode:
    """
    A node that applies face swapping using Deep Live Cam to frames in ComfyUI.
    """
    
    def __init__(self):
        self.face_swapper = None
        self.source_face = None
        self.initialized = False
        self.execution_providers = ["CPUExecutionProvider"]
        self.current_provider = None
        
        # Ensure models directory exists
        self.models_dir = os.path.join(dlc_path, "models")
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Download the required model if not present
        self._download_model()
    
    def _download_model(self):
        """Download the face swapping model if it doesn't exist."""
        model_path = os.path.join(self.models_dir, "inswapper_128_fp16.onnx")
        if not os.path.exists(model_path):
            logger.info("Downloading face swapper model...")
            conditional_download(
                self.models_dir,
                ["https://huggingface.co/hacksider/deep-live-cam/blob/main/inswapper_128_fp16.onnx"]
            )
            logger.info("Model downloaded successfully")
    
    def _init_face_swapper(self, execution_provider):
        """Initialize the face swapper module."""
        # Reset initialization if provider changes
        if self.current_provider != execution_provider:
            self.initialized = False
            self.current_provider = execution_provider
            
        if not self.initialized:
            try:
                # Update global execution providers
                dlc_globals.execution_providers = [execution_provider]
                self.execution_providers = [execution_provider]
                
                # Reset and initialize face swapper with new provider
                from .DeepLiveCam.modules.processors.frame import face_swapper
                face_swapper.FACE_SWAPPER = None
                
                # Explicitly set the model path in the face_swapper module
                face_swapper.models_dir = self.models_dir
                
                self.face_swapper = get_face_swapper()
                self.initialized = True
                logger.info(f"Face swapper initialized successfully with provider: {execution_provider}")
            except Exception as e:
                logger.error(f"Failed to initialize face swapper: {str(e)}")
                raise RuntimeError(f"Failed to initialize face swapper: {str(e)}")
    
    def _process_source_image(self, source_tensor):
        """Extract face from the source tensor image."""
        try:
            # Convert from tensor to numpy array (ComfyUI uses RGB format)
            if len(source_tensor.shape) == 4:
                source_np = source_tensor[0].cpu().numpy()
            else:
                source_np = source_tensor.cpu().numpy()
            
            # Convert from float [0,1] to uint8 [0,255] and RGB to BGR for OpenCV
            source_img = (source_np * 255).astype(np.uint8)
            source_img = cv2.cvtColor(source_img, cv2.COLOR_RGB2BGR)
            
            # Get the source face
            source_face = get_one_face(source_img)
            if source_face is None:
                raise ValueError("No face detected in source image")
            
            return source_face
        except Exception as e:
            logger.error(f"Error processing source face: {str(e)}")
            raise RuntimeError(f"Error processing source face: {str(e)}")
    
    @classmethod
    def INPUT_TYPES(cls):
        available_providers = get_available_execution_providers()
        provider_names = [p[0] for p in available_providers]
        
        return {
            "required": {
                "image": ("IMAGE",),
                "source_image": ("IMAGE",),
                "execution_provider": (provider_names, {"default": "CPUExecutionProvider"}),
                "many_faces": ("BOOLEAN", {"default": False}),
                "mouth_mask": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_image"
    CATEGORY = "face_swap"
    
    def process_image(self, image, source_image, execution_provider="CPUExecutionProvider", many_faces=False, mouth_mask=False):
        """
        Process the input image by applying face swapping.
        
        Args:
            image: Input image tensor from ComfyUI (RGB format)
            source_image: Source image tensor containing the face to swap
            execution_provider: ONNX Runtime execution provider
            many_faces: Whether to process all detected faces
            mouth_mask: Whether to apply mouth masking
        
        Returns:
            Processed image tensor
        """
        # Initialize face swapper if needed
        self._init_face_swapper(execution_provider)
        
        # Extract face from source image tensor
        self.source_face = self._process_source_image(source_image)
        
        # Check if source face is loaded
        if self.source_face is None:
            logger.warning("No source face loaded. Please provide a valid source image with a detectable face.")
            return (image,)
        
        # Convert from tensor to numpy array (ComfyUI uses RGB format)
        # First frame from batch if it's a batch
        if len(image.shape) == 4:
            image_np = image[0].cpu().numpy()
        else:
            image_np = image.cpu().numpy()
        
        # Convert from float [0,1] to uint8 [0,255] and RGB to BGR for OpenCV
        frame = (image_np * 255).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Set global variables for deep live cam
        dlc_globals.many_faces = many_faces
        dlc_globals.mouth_mask = mouth_mask
        
        # Process the frame
        if many_faces:
            target_faces = get_many_faces(frame)
            if target_faces:
                for target_face in target_faces:
                    frame = swap_face(self.source_face, target_face, frame)
        else:
            target_face = get_one_face(frame)
            if target_face:
                frame = swap_face(self.source_face, target_face, frame)
            else:
                logger.warning("No face detected in input frame")
        
        # Convert back to RGB for ComfyUI and normalize to [0,1]
        result = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        # Convert back to tensor, preserving the original shape
        if len(image.shape) == 4:
            # If input was batched, maintain batch dimension
            result_tensor = torch.from_numpy(result).unsqueeze(0).to(image.device)
        else:
            result_tensor = torch.from_numpy(result).to(image.device)
        
        return (result_tensor,)

# Node class mappings
NODE_CLASS_MAPPINGS = {
    "DeepLiveCamNode": DeepLiveCamNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DeepLiveCamNode": "Deep Live Cam Face Swap"
}