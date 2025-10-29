"""
Training Script với Heavy Data Augmentation (3x) - Multi-GPU Version
Tối ưu cho Kaggle với 2x GPU T4 (16GB VRAM mỗi GPU)

IMPROVEMENTS:
- Sử dụng torchrun để khởi tạo distributed training (như code 2)
- Tăng cường augmentation để sinh ra 3x dữ liệu gốc
- Cải thiện checkpoint management
- Validation trước training để phát hiện lỗi sớm

Usage:
    Single GPU: CUDA_LAUNCH_BLOCKING=1 python train_multi_gpu_3x_augmentation.py
    Multi GPU:  torchrun --nproc_per_node=2 train_multi_gpu_3x_augmentation.py
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
from PIL import Image
from torchvision import transforms
from typing import Optional, Dict, List, Tuple
import time
import warnings
warnings.filterwarnings('ignore')

# Import model
try:
    from model import ModernImageCaptioningModel
except ImportError:
    print("⚠️ Cannot import ModernImageCaptioningModel from model.py")
    print("   Please make sure model.py is in the same directory!")
    raise


# =====================================================================
# DISTRIBUTED TRAINING SETUP (theo code 2)
# =====================================================================
def setup_distributed():
    """Setup distributed training - Compatible with torchrun"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        
        if rank == 0:
            print(f"🚀 Distributed training initialized: {world_size} GPUs")
            for i in range(world_size):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        
        return local_rank, rank, world_size
    
    # Single GPU mode
    return 0, 0, 1


def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


# =====================================================================
# DATA PROCESSING
# =====================================================================
class SimpleFlickr8kProcessor:
    """Simplified data processor for Flickr8k"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / 'Images'
        self.captions_file = self.data_dir / 'captions.txt'
        
    def load_and_clean_captions(self):
        """Load captions from CSV"""
        df = pd.read_csv(self.captions_file)
        
        # Clean captions
        df['caption'] = df['caption'].str.lower()
        df['caption'] = df['caption'].str.replace(r'[^a-z\s]', '', regex=True)
        df['caption'] = df['caption'].str.strip()
        
        return df
    
    def build_vocabulary(self, df: pd.DataFrame, min_freq: int = 2) -> Dict:
        """Build vocabulary from captions"""
        word_freq = {}
        for caption in df['caption']:
            for word in caption.split():
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Filter by frequency
        vocab = [word for word, freq in word_freq.items() if freq >= min_freq]
        
        # Add special tokens
        special_tokens = ['<pad>', '<start>', '<end>', '<unk>']
        vocab = special_tokens + sorted(vocab)
        
        word2idx = {word: idx for idx, word in enumerate(vocab)}
        idx2word = {idx: word for word, idx in word2idx.items()}
        
        return word2idx, idx2word
    
    def split_dataset(self, df, train_ratio=0.8, val_ratio=0.1):
        """Split dataset"""
        unique_images = df['image'].unique()
        np.random.seed(42)
        np.random.shuffle(unique_images)
        
        n_train = int(len(unique_images) * train_ratio)
        n_val = int(len(unique_images) * val_ratio)
        
        train_images = unique_images[:n_train]
        val_images = unique_images[n_train:n_train + n_val]
        test_images = unique_images[n_train + n_val:]
        
        train_df = df[df['image'].isin(train_images)].reset_index(drop=True)
        val_df = df[df['image'].isin(val_images)].reset_index(drop=True)
        test_df = df[df['image'].isin(test_images)].reset_index(drop=True)
        
        return train_df, val_df, test_df


# =====================================================================
# SIMPLE DATASET (for validation - no augmentation)
# =====================================================================
class SimpleFlickr8kDataset(Dataset):
    """Simple dataset without augmentation (for validation)"""
    
    def __init__(
        self, 
        df: pd.DataFrame,
        images_dir: str,
        word2idx: Dict,
        max_length: int = 128,
        img_size: int = 224
    ):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.word2idx = word2idx
        self.max_length = max_length
        
        self.pad_idx = word2idx['<pad>']
        self.start_idx = word2idx['<start>']
        self.end_idx = word2idx['<end>']
        self.unk_idx = word2idx['<unk>']
        self.vocab_size = len(word2idx)
        
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def tokenize(self, caption: str) -> List[int]:
        """Tokenize caption to list of indices"""
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
        
        img_path = self.images_dir / row['image']
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            image = torch.zeros(3, 224, 224)
        
        tokens = self.tokenize(row['caption'])
        tokens = torch.tensor(tokens, dtype=torch.long)
        
        # Validation
        assert tokens.min() >= 0, f"Negative token: {tokens.min()}"
        assert tokens.max() < self.vocab_size, f"Token {tokens.max()} >= vocab_size {self.vocab_size}"
        
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
# HEAVY AUGMENTATION DATASETS (3 versions để tạo 3x data)
# =====================================================================
class AugmentedFlickr8kDataset(Dataset):
    """Dataset với augmentation để tạo thêm dữ liệu"""
    
    def __init__(
        self, 
        df: pd.DataFrame,
        images_dir: str,
        word2idx: Dict,
        max_length: int = 128,
        img_size: int = 224,
        augmentation_version: int = 1  # 1, 2, hoặc 3
    ):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.word2idx = word2idx
        self.max_length = max_length
        self.augmentation_version = augmentation_version
        
        self.pad_idx = word2idx['<pad>']
        self.start_idx = word2idx['<start>']
        self.end_idx = word2idx['<end>']
        self.unk_idx = word2idx['<unk>']
        self.vocab_size = len(word2idx)
        
        # Khác nhau về augmentation cho mỗi version
        if augmentation_version == 1:
            self.transform = self._get_augmentation_v1(img_size)
        elif augmentation_version == 2:
            self.transform = self._get_augmentation_v2(img_size)
        else:  # version 3
            self.transform = self._get_augmentation_v3(img_size)
    
    def _get_augmentation_v1(self, img_size):
        """Augmentation Version 1: Focus on geometric transforms"""
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(
                degrees=0, translate=(0.1, 0.1),
                scale=(0.9, 1.1), shear=5
            ),
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3,
                saturation=0.3, hue=0.1
            ),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _get_augmentation_v2(self, img_size):
        """Augmentation Version 2: Focus on color and blur"""
        return transforms.Compose([
            transforms.Resize((img_size + 48, img_size + 48)),
            transforms.RandomCrop((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=25),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4,
                saturation=0.4, hue=0.2
            ),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
            ], p=0.3),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _get_augmentation_v3(self, img_size):
        """Augmentation Version 3: Extreme/mixed augmentation"""
        return transforms.Compose([
            transforms.Resize((img_size + 64, img_size + 64)),
            transforms.RandomCrop((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=30),
            transforms.RandomAffine(
                degrees=0, translate=(0.15, 0.15),
                scale=(0.8, 1.2), shear=10
            ),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.ColorJitter(
                brightness=0.5, contrast=0.5,
                saturation=0.5, hue=0.25
            ),
            transforms.RandomGrayscale(p=0.15),
            transforms.RandomInvert(p=0.05),
            transforms.RandomPosterize(bits=4, p=0.1),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=7, sigma=(0.1, 3.0))
            ], p=0.4),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.4, scale=(0.02, 0.2)),
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
        
        img_path = self.images_dir / row['image']
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            image = torch.zeros(3, 224, 224)
        
        tokens = self.tokenize(row['caption'])
        tokens = torch.tensor(tokens, dtype=torch.long)
        
        # Validation
        assert tokens.min() >= 0, f"Negative token: {tokens.min()}"
        assert tokens.max() < self.vocab_size, f"Token {tokens.max()} >= vocab_size {self.vocab_size}"
        
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
# VALIDATION FUNCTION
# =====================================================================
def validate_dataloader(dataloader, vocab_size, num_batches=3, is_main=True):
    """Validate dataloader to detect issues early"""
    if not is_main:
        return
    
    print("\n" + "="*70)
    print("🔍 VALIDATING DATALOADER")
    print("="*70)
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
        
        input_ids = batch['input_ids']
        target_ids = batch['target_ids']
        
        print(f"\nBatch {batch_idx}:")
        print(f"  Input shape: {input_ids.shape}")
        print(f"  Target shape: {target_ids.shape}")
        print(f"  Input range: [{input_ids.min()}, {input_ids.max()}]")
        print(f"  Target range: [{target_ids.min()}, {target_ids.max()}]")
        print(f"  Vocab size: {vocab_size}")
        
        # Check for invalid tokens
        if input_ids.max() >= vocab_size:
            raise ValueError(f"⚠️ Input token {input_ids.max()} >= vocab_size {vocab_size}")
        
        if target_ids.max() >= vocab_size:
            raise ValueError(f"⚠️ Target token {target_ids.max()} >= vocab_size {vocab_size}")
        
        if input_ids.min() < 0:
            raise ValueError(f"⚠️ Negative input token: {input_ids.min()}")
        
        if target_ids.min() < 0:
            raise ValueError(f"⚠️ Negative target token: {target_ids.min()}")
        
        print("  ✓ Batch valid!")
    
    print("\n" + "="*70)
    print("✅ DATALOADER VALIDATION PASSED")
    print("="*70 + "\n")


# =====================================================================
# MULTI-GPU TRAINER (theo code 2 với cải tiến)
# =====================================================================
class MultiGPUTrainer:
    """Trainer hỗ trợ multi-GPU với DDP"""
    
    def __init__(
        self, 
        model: nn.Module, 
        train_loader: DataLoader, 
        val_loader: DataLoader,
        config: Dict, 
        device: str,
        local_rank: int,
        rank: int,
        world_size: int
    ):
        self.config = config
        self.device = device
        self.local_rank = local_rank
        self.rank = rank
        self.world_size = world_size
        self.is_main = (rank == 0)
        
        self.grad_accum_steps = config.get('grad_accum_steps', 4)
        self.grad_clip = config.get('grad_clip', 1.0)
        self.label_smoothing = config.get('label_smoothing', 0.1)
        self.use_amp = config.get('use_amp', True)
        
        self.patience = config.get('patience', 5)
        self.patience_counter = 0
        self.min_delta = config.get('min_delta', 0.001)
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Model setup
        self.model = model.to(device)
        if world_size > 1:
            self.model = DDP(
                self.model, 
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False
            )
        
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.scaler = GradScaler() if self.use_amp else None
        
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        self.checkpoint_dir = Path(config.get('checkpoint_dir', '/kaggle/working/checkpoints'))
        if self.is_main:
            self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    def _create_optimizer(self):
        """Create optimizer with proper weight decay"""
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
        
        # Set epoch for distributed sampler
        if hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(self.epoch)
        
        pbar = tqdm(self.train_loader, desc=f'[GPU{self.rank}] Epoch {self.epoch}',
                   disable=not self.is_main)
        
        for step, batch in enumerate(pbar):
            try:
                images = batch['image'].to(self.device, non_blocking=True)
                input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                target_ids = batch['target_ids'].to(self.device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
                
                # Forward with mixed precision
                with autocast(enabled=self.use_amp):
                    logits = self.model(images, input_ids)
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        target_ids.reshape(-1),
                        ignore_index=self.config.get('pad_idx', 0),
                        label_smoothing=self.label_smoothing
                    )
                    loss = loss / self.grad_accum_steps
                
                # Backward
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Update weights after gradient accumulation
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
                
                # Metrics
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
            
            except Exception as e:
                if self.is_main:
                    print(f"\n❌ Error at step {step}: {e}")
                raise
        
        # Aggregate metrics across GPUs
        if self.world_size > 1:
            metrics = torch.tensor([total_loss, total_tokens], device=self.device)
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            total_loss, total_tokens = metrics.cpu().numpy()
        
        avg_loss = total_loss / max(total_tokens, 1)
        return avg_loss
    
    @torch.no_grad()
    def validate(self):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        
        pbar = tqdm(self.val_loader, desc=f'[GPU{self.rank}] Validation',
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
        
        # Aggregate metrics across GPUs
        if self.world_size > 1:
            metrics = torch.tensor([total_loss, total_tokens], device=self.device)
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            total_loss, total_tokens = metrics.cpu().numpy()
        
        avg_loss = total_loss / max(total_tokens, 1)
        return avg_loss
    
    def check_early_stopping(self, val_loss):
        """Check early stopping condition"""
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            return False, True  # should_stop=False, is_best=True
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                return True, False  # should_stop=True, is_best=False
            return False, False
    
    def save_checkpoint(self, is_best: bool = False, is_last: bool = False):
        """Save checkpoint - only best and last"""
        if not self.is_main:
            return
        
        # Get model state (unwrap DDP if needed)
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
            print(f"🏆 Saved BEST model (val_loss={self.best_val_loss:.4f}): {best_path}")
        
        if is_last:
            last_path = self.checkpoint_dir / 'last_checkpoint_augmented.pt'
            torch.save(checkpoint, last_path)
            print(f"💾 Saved last checkpoint (epoch {self.epoch}): {last_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint to resume training"""
        if not os.path.exists(checkpoint_path):
            if self.is_main:
                print(f"⚠️ Checkpoint not found: {checkpoint_path}")
            return False
        
        if self.is_main:
            print(f"📥 Loading checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        if isinstance(self.model, DDP):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.patience_counter = checkpoint.get('patience_counter', 0)
        
        if self.is_main:
            print(f"✅ Resumed from epoch {self.epoch} (best_val_loss={self.best_val_loss:.4f})")
        
        return True
    
    def train(self, num_epochs: int):
        """Full training loop"""
        if self.is_main:
            print("\n" + "="*70)
            print("🚀 MULTI-GPU TRAINING với HEAVY AUGMENTATION (3x DATA)")
            print("="*70)
            
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            print(f"📊 Total parameters: {total_params/1e6:.2f}M")
            print(f"📊 Trainable parameters: {trainable_params/1e6:.2f}M")
            print(f"🖥️ World size: {self.world_size} GPUs")
            print(f"🔧 Gradient accumulation steps: {self.grad_accum_steps}")
            print(f"🔧 Effective batch size: {self.config['batch_size'] * self.grad_accum_steps * self.world_size}")
            print(f"🔧 Early stopping patience: {self.patience}")
            print(f"💾 Checkpoint strategy: Save BEST and LAST only")
            print("="*70 + "\n")
        
        start_epoch = self.epoch
        
        for epoch in range(start_epoch, start_epoch + num_epochs):
            self.epoch = epoch
            
            start_time = time.time()
            train_loss = self.train_epoch()
            train_time = time.time() - start_time
            
            val_loss = self.validate()
            
            if self.is_main:
                print(f"\n{'='*70}")
                print(f"📈 Epoch {epoch} Summary:")
                print(f"   Train Loss: {train_loss:.4f} | Train PPL: {np.exp(train_loss):.2f}")
                print(f"   Val Loss: {val_loss:.4f} | Val PPL: {np.exp(val_loss):.2f}")
                print(f"   Time: {train_time:.1f}s | Patience: {self.patience_counter}/{self.patience}")
                print(f"{'='*70}\n")
            
            # Early stopping check
            should_stop, is_best = self.check_early_stopping(val_loss)
            
            # Save checkpoints
            if is_best:
                self.save_checkpoint(is_best=True, is_last=False)
            
            # Always save last checkpoint
            self.save_checkpoint(is_best=False, is_last=True)
            
            if should_stop:
                if self.is_main:
                    print(f"\n🛑 Early stopping triggered at epoch {epoch}")
                    print(f"   Best validation loss: {self.best_val_loss:.4f}")
                break
        
        if self.is_main:
            print(f"\n✅ Training completed!")
            print(f"🏆 Best validation loss: {self.best_val_loss:.4f}")
            print(f"💾 Saved checkpoints:")
            print(f"   - best_model_augmented.pt")
            print(f"   - last_checkpoint_augmented.pt")


# =====================================================================
# MAIN FUNCTION
# =====================================================================
def main():
    """Main training script"""
    
    # Setup distributed training
    local_rank, rank, world_size = setup_distributed()
    device = f'cuda:{local_rank}'
    is_main = (rank == 0)
    
    # Configuration
    config = {
        # Paths (adjust for your Kaggle setup)
        'data_dir': '/kaggle/input/flickr8k',
        'vocab_dir': '/kaggle/input/vit-gpt2-nano/checkpoints',
        'checkpoint_dir': '/kaggle/working/checkpoints',
        'resume_checkpoint': None,  # Set to path if resuming
        
        # Model architecture (giữ nguyên như code 1)
        'vocab_size': None,  # Will be set after loading vocab
        'img_size': 224,
        'patch_size': 16,
        'max_seq_len': 128,
        'embed_dim': 768,
        'encoder_depth': 12,
        'decoder_depth': 12,
        'num_heads': 12,
        'num_kv_heads': 4,
        'mlp_ratio': 4.0,
        'dropout': 0.1,
        'drop_path_rate': 0.2,
        'num_registers': 4,
        
        # Training hyperparameters (giữ nguyên như code 1)
        'batch_size': 32,  # Per GPU
        'epochs': 2,
        'lr': 1e-4,
        'weight_decay': 0.02,
        'grad_clip': 1.0,
        'grad_accum_steps': 5,
        'label_smoothing': 0.15,
        
        # Early stopping (giữ nguyên như code 1)
        'patience': 5,
        'min_delta': 0.001,
        
        # Optimization (giữ nguyên như code 1)
        'use_amp': True,
        
        # Data (giữ nguyên như code 1)
        'num_workers': 2,  # Per GPU
        'min_word_freq': 2,
    }
    
    if is_main:
        print("\n" + "="*70)
        print("🎯 KHỞI TẠO TRAINING PIPELINE")
        print("="*70)
        print(f"🖥️ Device: {world_size}x {torch.cuda.get_device_name(local_rank)}")
        print(f"💾 VRAM per GPU: {torch.cuda.get_device_properties(local_rank).total_memory / 1e9:.1f} GB")
    
    # ===== STEP 1: LOAD OR BUILD VOCABULARY =====
    if is_main:
        print("\n" + "-"*70)
        print("STEP 1: Load/Build Vocabulary")
        print("-"*70)
    
    vocab_path = Path(config['vocab_dir']) / 'vocab.json'
    
    if vocab_path.exists():
        # Load existing vocabulary
        if is_main:
            print(f"📖 Loading vocabulary from {vocab_path}")
        with open(vocab_path, 'r') as f:
            vocab_data = json.load(f)
            word2idx = vocab_data['word2idx']
            idx2word = vocab_data['idx2word']
    else:
        # Build new vocabulary
        if is_main:
            print("🔨 Building new vocabulary...")
        processor = SimpleFlickr8kProcessor(config['data_dir'])
        df = processor.load_and_clean_captions()
        word2idx, idx2word = processor.build_vocabulary(df, min_freq=config['min_word_freq'])
        
        # Save vocabulary
        if is_main:
            vocab_save_path = Path(config['checkpoint_dir']) / 'vocab.json'
            vocab_save_path.parent.mkdir(exist_ok=True, parents=True)
            with open(vocab_save_path, 'w') as f:
                json.dump({'word2idx': word2idx, 'idx2word': idx2word}, f)
            print(f"💾 Saved vocabulary to {vocab_save_path}")
    
    config['vocab_size'] = len(word2idx)
    config['pad_idx'] = word2idx['<pad>']
    
    if is_main:
        print(f"✓ Vocabulary size: {config['vocab_size']} tokens")
    
    # ===== STEP 2: LOAD AND SPLIT DATA =====
    if is_main:
        print("\n" + "-"*70)
        print("STEP 2: Load and Split Dataset")
        print("-"*70)
    
    processor = SimpleFlickr8kProcessor(config['data_dir'])
    df = processor.load_and_clean_captions()
    train_df, val_df, test_df = processor.split_dataset(df)
    
    if is_main:
        print(f"✓ Train: {len(train_df)} captions from {train_df['image'].nunique()} images")
        print(f"✓ Val: {len(val_df)} captions from {val_df['image'].nunique()} images")
        print(f"✓ Test: {len(test_df)} captions from {test_df['image'].nunique()} images")
    
    images_dir = processor.images_dir
    
    # ===== STEP 3: CREATE 3x AUGMENTED DATASETS =====
    if is_main:
        print("\n" + "-"*70)
        print("STEP 3: Create 3x Augmented Training Datasets")
        print("-"*70)
    
    # Create 3 different augmented versions of training data
    train_dataset_v1 = AugmentedFlickr8kDataset(
        train_df, images_dir, word2idx,
        max_length=config['max_seq_len'],
        img_size=config['img_size'],
        augmentation_version=1
    )
    
    train_dataset_v2 = AugmentedFlickr8kDataset(
        train_df, images_dir, word2idx,
        max_length=config['max_seq_len'],
        img_size=config['img_size'],
        augmentation_version=2
    )
    
    train_dataset_v3 = AugmentedFlickr8kDataset(
        train_df, images_dir, word2idx,
        max_length=config['max_seq_len'],
        img_size=config['img_size'],
        augmentation_version=3
    )
    
    # Concatenate to create 3x data
    combined_train_dataset = ConcatDataset([
        train_dataset_v1,
        train_dataset_v2,
        train_dataset_v3
    ])
    
    if is_main:
        print(f"✓ Original training samples: {len(train_df)}")
        print(f"✓ Augmented training samples: {len(combined_train_dataset)} (3x)")
        print(f"  - Version 1 (Geometric): {len(train_dataset_v1)} samples")
        print(f"  - Version 2 (Color+Blur): {len(train_dataset_v2)} samples")
        print(f"  - Version 3 (Extreme): {len(train_dataset_v3)} samples")
    
    # Validation dataset (no augmentation)
    val_dataset = SimpleFlickr8kDataset(
        val_df, images_dir, word2idx,
        max_length=config['max_seq_len'],
        img_size=config['img_size']
    )
    
    if is_main:
        print(f"✓ Validation samples: {len(val_dataset)}")
    
    # ===== STEP 4: CREATE DATALOADERS =====
    if is_main:
        print("\n" + "-"*70)
        print("STEP 4: Create DataLoaders")
        print("-"*70)
    
    # Create distributed samplers
    train_sampler = DistributedSampler(
        combined_train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True
    ) if world_size > 1 else None
    
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    ) if world_size > 1 else None
    
    # Create dataloaders
    train_loader = DataLoader(
        combined_train_dataset,
        batch_size=config['batch_size'],
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=config['num_workers'],
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True if config['num_workers'] > 0 else False,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        sampler=val_sampler,
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True if config['num_workers'] > 0 else False
    )
    
    if is_main:
        print(f"✓ Train batches: {len(train_loader)}")
        print(f"✓ Val batches: {len(val_loader)}")
        print(f"✓ Effective batch size: {config['batch_size'] * world_size * config['grad_accum_steps']}")
    
    # ===== VALIDATE DATALOADER =====
    if is_main:
        validate_dataloader(train_loader, config['vocab_size'], num_batches=3, is_main=is_main)
    
    # ===== STEP 5: CREATE MODEL =====
    if is_main:
        print("\n" + "-"*70)
        print("STEP 5: Initialize Model")
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
    
    # ===== STEP 6: CREATE TRAINER =====
    if is_main:
        print("\n" + "-"*70)
        print("STEP 6: Initialize Trainer")
        print("-"*70)
    
    trainer = MultiGPUTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        local_rank=local_rank,
        rank=rank,
        world_size=world_size
    )
    
    # Load checkpoint if resuming
    if config.get('resume_checkpoint'):
        trainer.load_checkpoint(config['resume_checkpoint'])
    
    # ===== STEP 7: TRAIN =====
    if is_main:
        print("\n" + "-"*70)
        print("STEP 7: Start Training")
        print("-"*70)
    
    try:
        trainer.train(num_epochs=config['epochs'])
    except KeyboardInterrupt:
        if is_main:
            print("\n⚠️ Training interrupted by user")
            trainer.save_checkpoint(is_best=False, is_last=True)
    except Exception as e:
        if is_main:
            print(f"\n❌ Training failed with error: {e}")
            import traceback
            traceback.print_exc()
        raise
    finally:
        cleanup_distributed()
    
    if is_main:
        print("\n" + "="*70)
        print("✅ TRAINING PIPELINE COMPLETED")
        print("="*70)


if __name__ == '__main__':
    # Enable CUDA optimizations
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'
    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run training
    main()
