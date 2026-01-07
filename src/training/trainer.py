"""Training loop with AMP, gradient clipping, early stopping."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any, List, Optional, Union
import numpy as np
from pathlib import Path
import logging

from ..pipelines.sample import Sample
from ..tracking.base import BaseTracker
from .losses import LossRegistry

logger = logging.getLogger(__name__)

# Handle different PyTorch versions for AMP
# PyTorch 2.0+ uses torch.amp, older versions use torch.cuda.amp
HAS_AMP = False
AMP_NEW_API = False  # True if using torch.amp (new API with device_type param)

try:
    from torch.amp import GradScaler, autocast
    HAS_AMP = True
    AMP_NEW_API = True
except ImportError:
    try:
        from torch.cuda.amp import GradScaler, autocast
        HAS_AMP = True
        AMP_NEW_API = False
    except ImportError:
        HAS_AMP = False
        logger.warning("AMP not available in this PyTorch version")


class Trainer:
    """Training loop with AMP, gradient clipping, early stopping.
    
    Features:
    - Automatic mixed precision (AMP) for faster training on GPU
    - Gradient clipping for stability
    - CosineAnnealing learning rate schedule
    - Early stopping with patience
    - Automatic Sample → DataLoader conversion
    
    Example usage:
        >>> trainer = Trainer(model, config, tracker)
        >>> history = trainer.fit(train_samples, val_samples)
    """
    
    def __init__(self,
                 model: nn.Module,
                 config: Dict[str, Any],
                 tracker: Optional[BaseTracker] = None,
                 device: str = 'auto',
                 loss_config: Optional[Dict[str, Any]] = None,
                 verbose: bool = False):
        """Initialize the trainer.
        
        Args:
            model: PyTorch model
            config: Training configuration
            tracker: Experiment tracker
            device: Device to use ('auto', 'cuda', 'cpu')
            loss_config: Loss function configuration dict with 'name' key
                         and loss-specific parameters. If None, defaults to MSE.
            verbose: If True, print training progress every epoch
        """
        # Determine device
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        self.model = model.to(device)
        self.config = config
        self.tracker = tracker
        self.verbose = verbose
        
        # Loss - get from config or default to MSE
        if loss_config:
            loss_name = loss_config.get('name', 'mse')
            loss_params = {k: v for k, v in loss_config.items() if k != 'name'}
            self.criterion = LossRegistry.get(loss_name, **loss_params)
            logger.info(f"Using loss: {loss_name} with params: {loss_params}")
        else:
            self.criterion = LossRegistry.get('mse')
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.get('learning_rate', 1e-3),
            weight_decay=config.get('weight_decay', 0.01),
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.get('scheduler_T0', 50),
            T_mult=2,
        )
        
        # AMP
        self.use_amp = (
            config.get('use_amp', True) and 
            device == 'cuda' and 
            HAS_AMP
        )
        if self.use_amp:
            # Handle different PyTorch AMP API versions
            self.scaler = GradScaler('cuda') if AMP_NEW_API else GradScaler()
        else:
            self.scaler = None
        
        self.grad_clip = config.get('gradient_clip', 1.0)
        
        # Tracking
        self.best_val_loss = float('inf')
        self.best_state = None
        self.patience_counter = 0
        
        logger.info(f"Trainer initialized: device={device}, AMP={self.use_amp}")
    
    def _samples_to_loader(self, samples: List[Sample], 
                           batch_size: int, shuffle: bool) -> DataLoader:
        """Convert Sample list to DataLoader.
        
        Args:
            samples: List of Sample objects
            batch_size: Batch size
            shuffle: Whether to shuffle
        
        Returns:
            DataLoader
        """
        # Ensure samples are tensors
        for s in samples:
            s.to_tensor()
        
        # Check if sequences or static
        x_sample = samples[0].x
        if isinstance(x_sample, torch.Tensor):
            is_sequence = x_sample.dim() >= 2
        else:
            is_sequence = False
        
        if is_sequence:
            # Sequences - need to handle variable lengths
            X = torch.stack([s.x for s in samples])
            y = torch.stack([s.y for s in samples])
            
            if samples[0].t is not None:
                t = torch.stack([s.t for s in samples])
                dataset = TensorDataset(X, y, t)
            else:
                dataset = TensorDataset(X, y)
        else:
            # Static features
            X = torch.stack([s.x for s in samples])
            y = torch.stack([s.y for s in samples])
            dataset = TensorDataset(X, y)
        
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def fit(self,
            train_samples: List[Sample],
            val_samples: List[Sample],
            epochs: Optional[int] = None,
            patience: Optional[int] = None) -> Dict[str, List[float]]:
        """Train the model.
        
        Args:
            train_samples: Training samples
            val_samples: Validation samples
            epochs: Number of epochs (default from config)
            patience: Early stopping patience (default from config)
        
        Returns:
            Training history with train_loss and val_loss
        """
        epochs = epochs or self.config.get('epochs', 100)
        patience = patience or self.config.get('early_stopping_patience', 20)
        batch_size = self.config.get('batch_size', 32)
        
        train_loader = self._samples_to_loader(train_samples, batch_size, shuffle=True)
        val_loader = self._samples_to_loader(val_samples, batch_size, shuffle=False)
        
        history = {'train_loss': [], 'val_loss': [], 'lr': []}
        
        logger.info(f"Starting training: {epochs} epochs, batch_size={batch_size}")
        
        for epoch in range(epochs):
            # Train
            train_loss = self._train_epoch(train_loader)
            val_loss = self._validate(val_loader)
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['lr'].append(current_lr)
            
            # Log
            if self.tracker:
                self.tracker.log_metrics({
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'lr': current_lr,
                }, step=epoch)
            
            # Scheduler
            self.scheduler.step()
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Print progress
            if self.verbose:
                # Print every epoch if verbose, updating the same line
                status = f"  Epoch {epoch+1:3d}/{epochs} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | LR: {current_lr:.2e}"
                if val_loss < self.best_val_loss:
                    status += " ✓ (best)"
                else:
                    status += f" (patience: {self.patience_counter}/{patience})"
                print(f"\r{status}", end='', flush=True)
            elif (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} - Train: {train_loss:.5f}, Val: {val_loss:.5f}, LR: {current_lr:.2e}")
            
            if self.patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break
        
        # Print newline after training completes if verbose was enabled
        if self.verbose:
            print()  # Newline after final status update
        
        # Restore best
        if self.best_state:
            self.model.load_state_dict(self.best_state)
            logger.info(f"Restored best model with val_loss={self.best_val_loss:.5f}")
        
        return history
    
    def _train_epoch(self, loader: DataLoader) -> float:
        """Train for one epoch.
        
        Args:
            loader: Training data loader
        
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        
        for batch in loader:
            if len(batch) == 3:
                X, y, t = [b.to(self.device) for b in batch]
            else:
                X, y = [b.to(self.device) for b in batch]
                t = None
            
            self.optimizer.zero_grad(set_to_none=True)
            
            if self.use_amp:
                # Handle different PyTorch AMP API versions
                amp_context = autocast('cuda') if AMP_NEW_API else autocast()
                with amp_context:
                    pred = self.model(X, t=t)
                    loss = self.criterion(pred, y)
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pred = self.model(X, t=t)
                loss = self.criterion(pred, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(loader)
    
    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> float:
        """Validate the model.
        
        Args:
            loader: Validation data loader
        
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        
        for batch in loader:
            if len(batch) == 3:
                X, y, t = [b.to(self.device) for b in batch]
            else:
                X, y = [b.to(self.device) for b in batch]
                t = None
            
            if self.use_amp:
                # Handle different PyTorch AMP API versions
                amp_context = autocast('cuda') if AMP_NEW_API else autocast()
                with amp_context:
                    pred = self.model(X, t=t)
                    loss = self.criterion(pred, y)
            else:
                pred = self.model(X, t=t)
                loss = self.criterion(pred, y)
            
            total_loss += loss.item()
        
        return total_loss / len(loader) if len(loader) > 0 else 0.0
    
    @torch.no_grad()
    def predict(self, samples: List[Sample]) -> np.ndarray:
        """Get predictions for samples.
        
        Args:
            samples: List of samples
        
        Returns:
            Numpy array of predictions
        """
        self.model.eval()
        
        batch_size = self.config.get('batch_size', 32)
        loader = self._samples_to_loader(samples, batch_size, shuffle=False)
        
        predictions = []
        for batch in loader:
            if len(batch) == 3:
                X, _, t = [b.to(self.device) for b in batch]
            else:
                X, _ = [b.to(self.device) for b in batch]
                t = None
            
            pred = self.model(X, t=t)
            predictions.append(pred.cpu().numpy())
        
        return np.vstack(predictions)
    
    def save(self, path: Path) -> None:
        """Save model and optimizer state.
        
        Args:
            path: Path to save checkpoint
        """
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load(self, path: Path) -> None:
        """Load model and optimizer state.
        
        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        if 'scheduler_state' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        logger.info(f"Loaded checkpoint from {path}")
