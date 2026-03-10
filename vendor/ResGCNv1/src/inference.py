"""
ResGCN Inference Module

Unified inference pipeline that encapsulates:
- Model loading
- SkeletonDataConverter (JSON → tensor)
- SkeletonTransform (preprocessing)
- Prediction with probability distribution

Usage:
    inference = ResGCNInference.from_config(config, checkpoint_path)
    result = inference.predict(json_path)
    results = inference.predict_batch(json_paths)
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .preprocess.skeleton_converter import SkeletonDataConverter
from .dataset.skeleton_transform import SkeletonTransform


@dataclass
class PredictionOutput:
    """Single prediction result"""
    file: str
    predicted_age: float
    predicted_class: int
    confidence: float
    prob_distribution: List[float]
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            'file': self.file,
            'predicted_age': self.predicted_age,
            'predicted_class': self.predicted_class,
            'confidence': self.confidence,
            'prob_distribution': self.prob_distribution
        }
        if self.error:
            result['error'] = self.error
        return result


class ResGCNInference:
    """Unified ResGCN inference pipeline
    
    Encapsulates the complete inference flow:
    JSON → SkeletonDataConverter → SkeletonTransform → Model → PredictionOutput
    
    This ensures identical preprocessing as training (via SkeletonTransform).
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        skeleton_transform: SkeletonTransform,
        converter: SkeletonDataConverter,
        bin_centers: Optional[torch.Tensor] = None,
        use_ldl: bool = True
    ):
        """
        Args:
            model: Loaded ResGCN model
            device: torch device (cuda/cpu)
            skeleton_transform: Transform pipeline for preprocessing
            converter: JSON to tensor converter
            bin_centers: Optional bin centers for LDL prediction
            use_ldl: Whether to use Label Distribution Learning
        """
        self.model = model
        self.device = device
        self.skeleton_transform = skeleton_transform
        self.converter = converter
        self.bin_centers = bin_centers
        self.use_ldl = use_ldl
        
        self.model.eval()
    
    def predict(self, json_path: Union[str, Path]) -> PredictionOutput:
        """Predict from a single JSON file
        
        Args:
            json_path: Path to JSON skeleton file
            
        Returns:
            PredictionOutput with age, confidence, and distribution
        """
        json_path = str(json_path)
        
        if not os.path.exists(json_path):
            return PredictionOutput(
                file=json_path,
                predicted_age=0.0,
                predicted_class=0,
                confidence=0.0,
                prob_distribution=[],
                error='File not found'
            )
        
        try:
            # 1. JSON → Tensor (C, T, V, M)
            skeleton_data = self.converter.json_to_tensor(json_path)
            
            # 2. Transform (joint masking + multi_input) → (4, 6, T, V, M)
            data_numpy = self.skeleton_transform(skeleton_data)
            
            # 3. To device: (1, 4, 6, T, V, M)
            x = torch.tensor(data_numpy, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # 4. Inference
            with torch.no_grad():
                out, _ = self.model(x, None)
                probs = F.softmax(out, dim=1)
            
            probs_np = probs[0].cpu().numpy().tolist()
            predicted_class = int(np.argmax(probs_np))
            confidence = float(probs_np[predicted_class])
            
            # 5. Calculate predicted age
            if self.use_ldl and self.bin_centers is not None:
                pred_expectation = torch.sum(probs * self.bin_centers.to(self.device), dim=1)
                predicted_age = pred_expectation[0].item()
            else:
                if self.bin_centers is not None:
                    predicted_age = float(self.bin_centers[predicted_class].item())
                else:
                    predicted_age = float(predicted_class)
            
            return PredictionOutput(
                file=json_path,
                predicted_age=predicted_age,
                predicted_class=predicted_class,
                confidence=confidence,
                prob_distribution=probs_np
            )
            
        except Exception as e:
            logging.error(f'Error processing {json_path}: {e}')
            return PredictionOutput(
                file=json_path,
                predicted_age=0.0,
                predicted_class=0,
                confidence=0.0,
                prob_distribution=[],
                error=str(e)
            )
    
    def predict_batch(self, json_paths: List[Union[str, Path]]) -> List[PredictionOutput]:
        """Predict from multiple JSON files
        
        Args:
            json_paths: List of paths to JSON skeleton files
            
        Returns:
            List of PredictionOutput
        """
        results = []
        for path in json_paths:
            result = self.predict(path)
            results.append(result)
            
            if result.error is None:
                logging.info(f'Predicted {os.path.basename(str(path))}: '
                           f'age={result.predicted_age:.2f}, confidence={result.confidence:.3f}')
        
        return results
    
    @classmethod
    def from_processor(
        cls,
        processor,
        checkpoint_path: Optional[str] = None
    ) -> 'ResGCNInference':
        """Create inference from existing Processor instance
        
        Args:
            processor: Initialized Processor instance (has model, device, etc.)
            checkpoint_path: Optional path to load specific checkpoint
            
        Returns:
            ResGCNInference instance
        """
        from . import utils as U
        
        # Load checkpoint if specified
        if checkpoint_path and os.path.exists(checkpoint_path):
            logging.info(f'Loading from checkpoint: {checkpoint_path}')
            checkpoint = torch.load(checkpoint_path, map_location=processor.device, weights_only=False)
            state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
            
            # Handle DataParallel prefix
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            
            processor.model.module.load_state_dict(new_state_dict)
        else:
            # Load from work_dir
            checkpoint = U.load_checkpoint(processor.args.work_dir, processor.model_name)
            if checkpoint:
                processor.model.module.load_state_dict(checkpoint['model'])
        
        # Create converter
        _, _, max_frame, num_joint, num_person = processor.data_shape
        converter = SkeletonDataConverter(
            num_joint=num_joint,
            max_frame=max_frame,
            num_person=num_person
        )
        
        # Create transform
        skeleton_transform = SkeletonTransform.from_graph(
            processor.args.dataset,
            augmentor=None  # No augmentation for inference
        )
        
        return cls(
            model=processor.model,
            device=processor.device,
            skeleton_transform=skeleton_transform,
            converter=converter,
            bin_centers=processor.bin_centers,
            use_ldl=getattr(processor.args, 'use_ldl', False)
        )

    @classmethod
    def from_config(cls, args, checkpoint_path=None):
        """Create inference pipeline directly from configuration args
        
        This avoids Initializer/Processor overhead by setting up only
        components needed for inference.
        """
        from .initializer import Initializer
        from .preprocess.skeleton_converter import SkeletonDataConverter
        from .dataset.skeleton_transform import SkeletonTransform
        
        # Override checkpoint path if provided
        if checkpoint_path:
             args.pretrained_path = checkpoint_path
             
        # Use Initializer in inference_only mode to setup environment and model
        initializer = Initializer(args, None, inference_only=True)
        
        # Create Converter
        _, _, max_frame, num_joint, num_person = initializer.data_shape
        converter = SkeletonDataConverter(
            num_joint=num_joint,
            max_frame=max_frame,
            num_person=num_person
        )
        
        # Create Transform
        skeleton_transform = SkeletonTransform.from_graph(
            args.dataset,
            augmentor=None
        )
        
        return cls(
            model=initializer.model,
            device=initializer.device,
            skeleton_transform=skeleton_transform,
            converter=converter,
            bin_centers=initializer.bin_centers,
            use_ldl=getattr(args, 'use_ldl', False)
        )
