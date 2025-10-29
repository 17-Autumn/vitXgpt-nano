"""
Resume Training với Heavy Data Augmentation để giảm Overfitting
- Load checkpoint cuối cùng
- Tăng dữ liệu gấp 3 lần với augmentation mạnh
- Tiếp tục training với early stopping

Usage:
    python train_with_augmentation.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler

import os
import sys
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageFilter
from torchvision import transforms
from typing import Optional, Dict, List, Tuple
import time
import warnings
warnings.filterwarnings('ignore')

from model import ModernImageCaptioningModel


# =====================================================================
# HEAVY DATA AUGMENTATION DATASET
# =====================================================================
class HeavyAugmentedFlickr8kDataset(Dataset):
    """
    Dataset với augmentation CỰC MẠNH để chống overfitting
    - Blur nhiều mức độ
    - Rotation ngẫu nhiên
    - Flip horizontal + vertical
    - Color jitter mạnh
    - Random erasing
    - Gaussian noise
    """
    
    def __init__(
        self, 
        df: pd.DataFrame,
        images_dir: str,
        word2idx: Dict,
        max_length: int = 128,
        img_size: int = 224,
        augmentation_level: str = 'heavy'  # 'heavy', 'extreme'
    ):
        self.df = df
        self.images_dir = Path(images_dir)
        self.word2idx = word2idx
        self.max_length = max_length
        self.augmentation_level = augmentation_level
        
        # Cache special token indices
        self.pad_idx = word2idx['<pad>']
        self.start_idx = word2idx['<start>']
        self.end_idx = word2idx['<end>']
        self.unk_idx = word2idx['<unk>']
        self.vocab_size = len(word2idx)
        
        # Heavy augmentation transforms
        if augmentation_level == 'heavy':
            self.transform = self._get_heavy_augmentation(img_size)
        else:  # extreme
            self.transform = self._get_extreme_augmentation(img_size)
    
    def _get_heavy_augmentation(self, img_size):
        """Augmentation mạnh nhưng vẫn giữ tính realism"""
        return transforms.Compose([
            transforms.Resize((img_size + 48, img_size + 48), 
                            interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomCrop((img_size, img_size)),
            
            # Geometric augmentations
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),  # Thêm vertical flip
            transforms.RandomRotation(degrees=30),  # Tăng từ 10 lên 30 độ
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),  # Translation
                scale=(0.8, 1.2),      # Scaling
                shear=10               # Shearing
            ),
            
            # Color augmentations
            transforms.ColorJitter(
                brightness=0.4,  # Tăng từ 0.2
                contrast=0.4,    # Tăng từ 0.2
                saturation=0.4,  # Tăng từ 0.2
                hue=0.2          # Tăng từ 0.1
            ),
            transforms.RandomGrayscale(p=0.1),  # Thêm grayscale
            
            # Blur augmentations
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
            ], p=0.3),
            
            transforms.ToTensor(),
            
            # Random erasing (giống cutout)
            transforms.RandomErasing(
                p=0.3,
                scale=(0.02, 0.15),
                ratio=(0.3, 3.3)
            ),
            
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _get_extreme_augmentation(self, img_size):
        """Augmentation cực mạnh cho dữ liệu augmented thêm"""
        return transforms.Compose([
            transforms.Resize((img_size + 64, img_size + 64), 
                            interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomCrop((img_size, img_size)),
            
            # Extreme geometric
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=45),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.15, 0.15),
                scale=(0.7, 1.3),
                shear=15
            ),
            transforms.RandomPerspective(distortion_scale=0.3, p=0.3),
            
            # Extreme color
            transforms.ColorJitter(
                brightness=0.5,
                contrast=0.5,
                saturation=0.5,
                hue=0.3
            ),
            transforms.RandomGrayscale(p=0.15),
            transforms.RandomInvert(p=0.1),
            transforms.RandomPosterize(bits=4, p=0.2),
            transforms.RandomSolarize(threshold=128, p=0.2),
            
            # Multiple blur options
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=7, sigma=(0.1, 3.0))
            ], p=0.4),
            
            transforms.ToTensor(),
            
            # Aggressive erasing
            transforms.RandomErasing(
                p=0.4,
                scale=(0.02, 0.2),
                ratio=(0.3, 3.3)
            ),
            
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def tokenize(self, caption: str) -> List[int]:
        """Tokenize caption"""
        words = caption.split()
        tokens = [self.word2idx.get(word, self.unk_idx) for word in words]
        tokens = [self.start_idx] + tokens + [self.end_idx]
        
        if len(tokens) > self.max_length:
            tokens = [self.start_idx] + tokens[1:self.max_length-1] + [self.end_idx]
        
        if len(tokens) < self.max_length:
            tokens = tokens + [self.pad_idx] * (self.max_length - len(tokens))
        
        return tokens
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        img_path = self.images_dir / row['image']
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            print(f"⚠️ Error loading {img_path}: {e}")
            image = torch.zeros(3, 224, 224)
        
        # Tokenize
        tokens = self.tokenize(row['caption'])
        tokens = torch.tensor(tokens, dtype=torch.long)
        
        input_ids = tokens[:-1]
        target_ids = tokens[1:]
        attention_mask = (target_ids != self.pad_idx).long()
        
        return {
            'image': image,
            'input_ids': input_ids,
            'target_ids': target_ids,
            'attention_mask': attention_mask,
            'caption': row['caption']
        }


# =====================================================================
# ENHANCED TRAINER với Early Stopping
# =====================================================================
class EnhancedTrainer:
    """
    Trainer với:
    - Early stopping
    - Learning rate reduction on plateau
    - Better checkpoint management
    """
    
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                 config: Dict, device: str = 'cuda', local_rank: int = 0):
        self.config = config
        self.device = device
        self.local_rank = local_rank
        self.is_main = (local_rank == 0)
        
        # Training hyperparameters
        self.grad_accum_steps = config.get('grad_accum_steps', 4)
        self.grad_clip = config.get('grad_clip', 1.0)
        self.label_smoothing = config.get('label_smoothing', 0.1)
        self.use_amp = config.get('use_amp', True)
        
        # Early stopping parameters
        self.patience = config.get('patience', 5)
        self.patience_counter = 0
        self.min_delta = config.get('min_delta', 0.001)
        
        # Data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Model setup
        self.model = model.to(device)
        if torch.cuda.device_count() > 1 and dist.is_initialized():
            self.model = DDP(self.model, device_ids=[local_rank])
        
        # Optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Mixed precision
        self.scaler = GradScaler() if self.use_amp else None
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # Checkpoint directory
        self.checkpoint_dir = Path(config.get('checkpoint_dir', './checkpoints'))
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    def _create_optimizer(self):
        """Create optimizer with separate weight decay"""
        decay = []
        no_decay = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'bias' in name or 'norm' in name or 'embed' in name:
                no_decay.append(param)
            else:
                decay.append(param)
        
        optimizer = AdamW([
            {'params': decay, 'weight_decay': self.config.get('weight_decay', 0.01)},
            {'params': no_decay, 'weight_decay': 0.0}
        ], lr=self.config.get('lr', 3e-4), betas=(0.9, 0.95), eps=1e-8)
        
        return optimizer
    
    def _create_scheduler(self):
        """Create OneCycleLR scheduler"""
        total_steps = len(self.train_loader) * self.config.get('epochs', 20) // self.grad_accum_steps
        
        scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.config.get('lr', 3e-4),
            total_steps=total_steps,
            pct_start=0.05,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=1e4
        )
        
        return scheduler
    
    def train_epoch(self):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        total_tokens = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch}', 
                   disable=not self.is_main)
        
        for step, batch in enumerate(pbar):
            images = batch['image'].to(self.device, non_blocking=True)
            input_ids = batch['input_ids'].to(self.device, non_blocking=True)
            target_ids = batch['target_ids'].to(self.device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
            
            with autocast(enabled=self.use_amp):
                logits = self.model(images, input_ids)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    target_ids.reshape(-1),
                    ignore_index=self.config.get('pad_idx', 0),
                    label_smoothing=self.label_smoothing
                )
                loss = loss / self.grad_accum_steps
            
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            if (step + 1) % self.grad_accum_steps == 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
                self.global_step += 1
            
            batch_loss = loss.item() * self.grad_accum_steps
            batch_tokens = attention_mask.sum().item()
            total_loss += batch_loss * batch_tokens
            total_tokens += batch_tokens
            
            if self.is_main:
                pbar.set_postfix({
                    'loss': f'{batch_loss:.4f}',
                    'ppl': f'{np.exp(batch_loss):.2f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                })
        
        avg_loss = total_loss / max(total_tokens, 1)
        return avg_loss
    
    @torch.no_grad()
    def validate(self):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        
        pbar = tqdm(self.val_loader, desc='Validation', 
                   disable=not self.is_main)
        
        for batch in pbar:
            images = batch['image'].to(self.device, non_blocking=True)
            input_ids = batch['input_ids'].to(self.device, non_blocking=True)
            target_ids = batch['target_ids'].to(self.device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
            
            with autocast(enabled=self.use_amp):
                logits = self.model(images, input_ids)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    target_ids.reshape(-1),
                    ignore_index=self.config.get('pad_idx', 0)
                )
            
            batch_tokens = attention_mask.sum().item()
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens
            
            if self.is_main:
                pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / max(total_tokens, 1)
        return avg_loss
    
    def check_early_stopping(self, val_loss):
        """Check if should early stop"""
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            return False, True  # Continue training, is_best
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                return True, False  # Stop training, not_best
            return False, False  # Continue training, not_best
    
    def save_checkpoint(self, is_best: bool = False, is_last: bool = False):
        """Save checkpoint"""
        if not self.is_main:
            return
        
        if isinstance(self.model, DDP):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'patience_counter': self.patience_counter
        }
        
        if is_best:
            best_path = self.checkpoint_dir / 'best_model_augmented.pt'
            torch.save(checkpoint, best_path)
            print(f"🏆 Saved BEST augmented model (val_loss={self.best_val_loss:.4f}): {best_path}")
        
        if is_last:
            last_path = self.checkpoint_dir / 'last_checkpoint_augmented.pt'
            torch.save(checkpoint, last_path)
            print(f"💾 Saved last checkpoint (epoch {self.epoch}): {last_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint to resume training"""
        if not os.path.exists(checkpoint_path):
            print(f"⚠️ Checkpoint not found: {checkpoint_path}")
            return False
        
        print(f"📥 Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model
        if isinstance(self.model, DDP):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer and scheduler
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        self.epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.patience_counter = checkpoint.get('patience_counter', 0)
        
        print(f"✅ Resumed from epoch {self.epoch}, best_val_loss={self.best_val_loss:.4f}")
        return True
    
    def train(self, num_epochs: int):
        """Full training loop with early stopping"""
        print("\n" + "="*70)
        print("🚀 BẮT ĐẦU TRAINING VỚI HEAVY AUGMENTATION")
        print("="*70)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        if self.is_main:
            print(f"📊 Total parameters: {total_params/1e6:.2f}M")
            print(f"📊 Trainable parameters: {trainable_params/1e6:.2f}M")
            print(f"🔧 Gradient accumulation: {self.grad_accum_steps}")
            print(f"🔧 Early stopping patience: {self.patience}")
            print(f"💾 Data augmentation: HEAVY (3x augmented dataset)")
            print("="*70 + "\n")
        
        start_epoch = self.epoch
        
        for epoch in range(start_epoch, start_epoch + num_epochs):
            self.epoch = epoch
            
            if hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)
            
            # Train
            start_time = time.time()
            train_loss = self.train_epoch()
            train_time = time.time() - start_time
            
            # Validate
            val_loss = self.validate()
            
            if self.is_main:
                print(f"\n{'='*70}")
                print(f"📈 Epoch {epoch} Summary:")
                print(f"   Train Loss: {train_loss:.4f} | Train PPL: {np.exp(train_loss):.2f}")
                print(f"   Val Loss: {val_loss:.4f} | Val PPL: {np.exp(val_loss):.2f}")
                print(f"   Time: {train_time:.1f}s")
                print(f"   Patience counter: {self.patience_counter}/{self.patience}")
                print(f"{'='*70}\n")
            
            # Early stopping check
            should_stop, is_best = self.check_early_stopping(val_loss)
            
            if is_best:
                self.save_checkpoint(is_best=True, is_last=False)
            
            self.save_checkpoint(is_best=False, is_last=True)
            
            if should_stop:
                if self.is_main:
                    print("\n" + "="*70)
                    print("🛑 EARLY STOPPING TRIGGERED")
                    print(f"   No improvement for {self.patience} epochs")
                    print(f"   Best val loss: {self.best_val_loss:.4f}")
                    print("="*70)
                break
        
        if self.is_main:
            print("\n" + "="*70)
            print("✅ TRAINING COMPLETED")
            print(f"🏆 Best validation loss: {self.best_val_loss:.4f}")
            print("="*70)


# =====================================================================
# MAIN FUNCTION
# =====================================================================
def main():
    """Main training script with heavy augmentation"""
    
    config = {
        # Paths
        'data_dir': '/kaggle/input/flickr8k',
        'checkpoint_dir': '/kaggle/input/vit-gpt2-nano/checkpoints',
        'resume_checkpoint': '/kaggle/input/vit-gpt2-nano/checkpoints/last_checkpoint.pt',
        
        # Model architecture (same as before)
        'vocab_size': None,
        'img_size': 224,
        'patch_size': 16,
        'max_seq_len': 128,
        'embed_dim': 768,
        'encoder_depth': 12,
        'decoder_depth': 12,
        'num_heads': 12,
        'num_kv_heads': 4,
        'mlp_ratio': 4.0,
        'dropout': 0.1,  # Tăng dropout
        'drop_path_rate': 0.2,  # Tăng drop path
        'num_registers': 4,
        
        # Training (adjusted for augmented data)
        'batch_size': 36,  # Giảm một chút do augmentation nặng
        'epochs': 5,  # Training thêm
        'lr': 1e-4,  # Lower learning rate khi resume
        'weight_decay': 0.02,  # Tăng weight decay
        'grad_clip': 1.0,
        'grad_accum_steps': 5,  # Tăng để maintain effective batch size
        'label_smoothing': 0.15,  # Tăng label smoothing
        
        # Early stopping
        'patience': 5,
        'min_delta': 0.001,
        
        # Optimization
        'use_amp': True,
        
        # Data
        'num_workers': 4,
        'pin_memory': True,
        'prefetch_factor': 2,
        'min_word_freq': 2,
        
        # Augmentation
        'augmentation_multiplier': 3,  # 3x data
    }
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    print("\n" + "="*70)
    print("🎯 RESUME TRAINING VỚI HEAVY DATA AUGMENTATION")
    print("="*70)
    print(f"🖥️ Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    # ===== LOAD DATA =====
    print("\n" + "-"*70)
    print("STEP 1: Load Data và Vocabulary")
    print("-"*70)
    
    # Load vocabulary
    vocab_path = Path(config['checkpoint_dir']) / 'vocab.json'
    with open(vocab_path, 'r') as f:
        vocab_data = json.load(f)
        word2idx = vocab_data['word2idx']
        idx2word = vocab_data['idx2word']
    
    config['vocab_size'] = len(word2idx)
    config['pad_idx'] = word2idx['<pad>']
    
    print(f"✓ Loaded vocabulary: {config['vocab_size']} tokens")
    
    # Load captions
    from train import Flickr8kProcessor
    processor = Flickr8kProcessor(config['data_dir'])
    df = processor.load_and_clean_captions()
    train_df, val_df, test_df = processor.split_dataset(df)
    
    # ===== CREATE AUGMENTED DATASETS =====
    print("\n" + "-"*70)
    print("STEP 2: Create 3x Augmented Training Dataset")
    print("-"*70)
    
    images_dir = processor.images_dir
    
    # Dataset 1: Heavy augmentation
    train_dataset_1 = HeavyAugmentedFlickr8kDataset(
        train_df, images_dir, word2idx,
        max_length=config['max_seq_len'],
        img_size=config['img_size'],
        augmentation_level='heavy'
    )
    
    # Dataset 2: Heavy augmentation (different random seed)
    train_dataset_2 = HeavyAugmentedFlickr8kDataset(
        train_df, images_dir, word2idx,
        max_length=config['max_seq_len'],
        img_size=config['img_size'],
        augmentation_level='heavy'
    )
    
    # Dataset 3: Extreme augmentation
    train_dataset_3 = HeavyAugmentedFlickr8kDataset(
        train_df, images_dir, word2idx,
        max_length=config['max_seq_len'],
        img_size=config['img_size'],
        augmentation_level='extreme'
    )
    
    # Combine all datasets
    combined_train_dataset = ConcatDataset([
        train_dataset_1,
        train_dataset_2,
        train_dataset_3
    ])
    
    print(f"✓ Original training samples: {len(train_dataset_1)}")
    print(f"✓ Augmented training samples: {len(combined_train_dataset)} (3x)")
    
    # Validation dataset (no augmentation)
    from train import Flickr8kDataset
    val_dataset = Flickr8kDataset(
        val_df, images_dir, word2idx,
        max_length=config['max_seq_len'],
        img_size=config['img_size'],
        is_train=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        combined_train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
        prefetch_factor=config['prefetch_factor'],
        persistent_workers=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
        prefetch_factor=config['prefetch_factor'],
        persistent_workers=True
    )
    
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")
    
    # ===== CREATE MODEL =====
    print("\n" + "-"*70)
    print("STEP 3: Create Model")
    print("-"*70)
    
    model = ModernImageCaptioningModel(
        vocab_size=config['vocab_size'],
        img_size=config['img_size'],
        patch_size=config['patch_size'],
        max_seq_len=config['max_seq_len'],
        embed_dim=config['embed_dim'],
        encoder_depth=config['encoder_depth'],
        decoder_depth=config['decoder_depth'],
        num_heads=config['num_heads'],
        num_kv_heads=config['num_kv_heads'],
        mlp_ratio=config['mlp_ratio'],
        dropout=config['dropout'],
        drop_path_rate=config['drop_path_rate'],
        num_registers=config['num_registers']
    )
    
    # ===== CREATE TRAINER AND LOAD CHECKPOINT =====
    print("\n" + "-"*70)
    print("STEP 4: Create Trainer và Load Checkpoint")
    print("-"*70)
    
    trainer = EnhancedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    # Load previous checkpoint
    checkpoint_loaded = trainer.load_checkpoint(config['resume_checkpoint'])
    
    if not checkpoint_loaded:
        print("⚠️ Could not load checkpoint, starting from scratch")
    
    # ===== TRAIN =====
    print("\n" + "-"*70)
    print("STEP 5: Start Training with Heavy Augmentation")
    print("-"*70)
    
    trainer.train(num_epochs=config['epochs'])
    
    print("\n" + "="*70)
    print("✅ ALL DONE!")
    print("="*70)


if __name__ == '__main__':
    # Enable optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    
    # Set seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run
    main()
