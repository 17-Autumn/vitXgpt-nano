"""
Resume Training với Heavy Data Augmentation - Multi-GPU Version (FIXED)
Hỗ trợ training trên 2 GPU T4 trên Kaggle

Usage:
    python train_with_augmentation_multi_gpu_fixed.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import torch.distributed as dist
import torch.multiprocessing as mp
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

# Import model - MAKE SURE model.py is available
try:
    from model import ModernImageCaptioningModel
except ImportError:
    print("⚠️ Cannot import ModernImageCaptioningModel from model.py")
    print("   Please make sure model.py is in the same directory!")
    raise


# =====================================================================
# DISTRIBUTED TRAINING SETUP
# =====================================================================
def setup_distributed(rank, world_size):
    """Khởi tạo distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Khởi tạo process group
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    
    torch.cuda.set_device(rank)
    
    if rank == 0:
        print(f"🚀 Process {rank}/{world_size} initialized on GPU {rank}")


def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


# =====================================================================
# DATA PROCESSING (Simplified Flickr8k Processor)
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
# SIMPLE DATASET (for validation)
# =====================================================================
class SimpleFlickr8kDataset(Dataset):
    """Simple dataset without heavy augmentation (for validation)"""
    
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
        
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def tokenize(self, caption: str) -> List[int]:
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
# HEAVY AUGMENTATION DATASET
# =====================================================================
class HeavyAugmentedFlickr8kDataset(Dataset):
    """Dataset với augmentation CỰC MẠNH"""
    
    def __init__(
        self, 
        df: pd.DataFrame,
        images_dir: str,
        word2idx: Dict,
        max_length: int = 128,
        img_size: int = 224,
        augmentation_level: str = 'heavy'
    ):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.word2idx = word2idx
        self.max_length = max_length
        self.augmentation_level = augmentation_level
        
        self.pad_idx = word2idx['<pad>']
        self.start_idx = word2idx['<start>']
        self.end_idx = word2idx['<end>']
        self.unk_idx = word2idx['<unk>']
        
        if augmentation_level == 'heavy':
            self.transform = self._get_heavy_augmentation(img_size)
        else:
            self.transform = self._get_extreme_augmentation(img_size)
    
    def _get_heavy_augmentation(self, img_size):
        """Heavy augmentation"""
        return transforms.Compose([
            transforms.Resize((img_size + 48, img_size + 48)),
            transforms.RandomCrop((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=30),
            transforms.RandomAffine(
                degrees=0, translate=(0.1, 0.1),
                scale=(0.8, 1.2), shear=10
            ),
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
    
    def _get_extreme_augmentation(self, img_size):
        """Extreme augmentation"""
        return transforms.Compose([
            transforms.Resize((img_size + 64, img_size + 64)),
            transforms.RandomCrop((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=45),
            transforms.RandomAffine(
                degrees=0, translate=(0.15, 0.15),
                scale=(0.7, 1.3), shear=15
            ),
            transforms.RandomPerspective(distortion_scale=0.3, p=0.3),
            transforms.ColorJitter(
                brightness=0.5, contrast=0.5,
                saturation=0.5, hue=0.3
            ),
            transforms.RandomGrayscale(p=0.15),
            transforms.RandomInvert(p=0.1),
            transforms.RandomPosterize(bits=4, p=0.2),
            transforms.RandomSolarize(threshold=128, p=0.2),
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
# MULTI-GPU TRAINER
# =====================================================================
class MultiGPUTrainer:
    """Trainer hỗ trợ multi-GPU với DDP"""
    
    def __init__(
        self, 
        model: nn.Module, 
        train_loader: DataLoader, 
        val_loader: DataLoader,
        config: Dict, 
        rank: int,
        world_size: int
    ):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.is_main = (rank == 0)
        
        self.device = f'cuda:{rank}'
        torch.cuda.set_device(rank)
        
        self.grad_accum_steps = config.get('grad_accum_steps', 4)
        self.grad_clip = config.get('grad_clip', 1.0)
        self.label_smoothing = config.get('label_smoothing', 0.1)
        self.use_amp = config.get('use_amp', True)
        
        self.patience = config.get('patience', 5)
        self.patience_counter = 0
        self.min_delta = config.get('min_delta', 0.001)
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.model = model.to(self.device)
        self.model = DDP(
            self.model, 
            device_ids=[rank],
            output_device=rank,
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
        self.model.train()
        total_loss = 0
        total_tokens = 0
        
        if hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(self.epoch)
        
        pbar = tqdm(self.train_loader, desc=f'[GPU{self.rank}] Epoch {self.epoch}',
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
        
        metrics = torch.tensor([total_loss, total_tokens], device=self.device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        total_loss, total_tokens = metrics.cpu().numpy()
        
        avg_loss = total_loss / max(total_tokens, 1)
        return avg_loss
    
    @torch.no_grad()
    def validate(self):
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
        
        metrics = torch.tensor([total_loss, total_tokens], device=self.device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        total_loss, total_tokens = metrics.cpu().numpy()
        
        avg_loss = total_loss / max(total_tokens, 1)
        return avg_loss
    
    def check_early_stopping(self, val_loss):
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            return False, True
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                return True, False
            return False, False
    
    def save_checkpoint(self, is_best: bool = False, is_last: bool = False):
        if not self.is_main:
            return
        
        model_state = self.model.module.state_dict()
        
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
            print(f"🏆 Saved BEST model: {best_path}")
        
        if is_last:
            last_path = self.checkpoint_dir / 'last_checkpoint_augmented.pt'
            torch.save(checkpoint, last_path)
            print(f"💾 Saved last checkpoint: {last_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        if not os.path.exists(checkpoint_path):
            if self.is_main:
                print(f"⚠️ Checkpoint not found: {checkpoint_path}")
            return False
        
        if self.is_main:
            print(f"📥 Loading checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.module.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.patience_counter = checkpoint.get('patience_counter', 0)
        
        if self.is_main:
            print(f"✅ Resumed from epoch {self.epoch}")
        
        return True
    
    def train(self, num_epochs: int):
        if self.is_main:
            print("\n" + "="*70)
            print("🚀 MULTI-GPU TRAINING với HEAVY AUGMENTATION")
            print("="*70)
            print(f"🖥️ World size: {self.world_size} GPUs")
            print(f"🔧 Early stopping patience: {self.patience}")
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
                print(f"📈 Epoch {epoch}:")
                print(f"   Train Loss: {train_loss:.4f} | PPL: {np.exp(train_loss):.2f}")
                print(f"   Val Loss: {val_loss:.4f} | PPL: {np.exp(val_loss):.2f}")
                print(f"   Time: {train_time:.1f}s | Patience: {self.patience_counter}/{self.patience}")
                print(f"{'='*70}\n")
            
            should_stop, is_best = self.check_early_stopping(val_loss)
            
            if is_best:
                self.save_checkpoint(is_best=True)
            
            self.save_checkpoint(is_last=True)
            
            if should_stop:
                if self.is_main:
                    print(f"\n🛑 Early stopping at epoch {epoch}")
                break
        
        if self.is_main:
            print(f"\n✅ Training completed! Best val loss: {self.best_val_loss:.4f}")


# =====================================================================
# WORKER FUNCTION
# =====================================================================
def train_worker(rank, world_size, config):
    """Worker function cho mỗi GPU"""
    
    setup_distributed(rank, world_size)
    
    try:
        torch.manual_seed(42 + rank)
        np.random.seed(42 + rank)
        
        # Load vocabulary
        vocab_path = Path(config['vocab_dir']) / 'vocab.json'
        with open(vocab_path, 'r') as f:
            vocab_data = json.load(f)
            word2idx = vocab_data['word2idx']
        
        config['vocab_size'] = len(word2idx)
        config['pad_idx'] = word2idx['<pad>']
        
        if rank == 0:
            print(f"✓ Loaded vocabulary: {config['vocab_size']} tokens")
        
        # Load data
        processor = SimpleFlickr8kProcessor(config['data_dir'])
        df = processor.load_and_clean_captions()
        train_df, val_df, _ = processor.split_dataset(df)
        
        images_dir = processor.images_dir
        
        # Create augmented datasets
        train_dataset_1 = HeavyAugmentedFlickr8kDataset(
            train_df, images_dir, word2idx,
            max_length=config['max_seq_len'],
            img_size=config['img_size'],
            augmentation_level='heavy'
        )
        
        train_dataset_2 = HeavyAugmentedFlickr8kDataset(
            train_df, images_dir, word2idx,
            max_length=config['max_seq_len'],
            img_size=config['img_size'],
            augmentation_level='heavy'
        )
        
        train_dataset_3 = HeavyAugmentedFlickr8kDataset(
            train_df, images_dir, word2idx,
            max_length=config['max_seq_len'],
            img_size=config['img_size'],
            augmentation_level='extreme'
        )
        
        combined_train_dataset = ConcatDataset([
            train_dataset_1,
            train_dataset_2,
            train_dataset_3
        ])
        
        if rank == 0:
            print(f"✓ Augmented training samples: {len(combined_train_dataset)} (3x)")
        
        # Validation dataset
        val_dataset = SimpleFlickr8kDataset(
            val_df, images_dir, word2idx,
            max_length=config['max_seq_len'],
            img_size=config['img_size']
        )
        
        # Create samplers
        train_sampler = DistributedSampler(
            combined_train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True
        )
        
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            combined_train_dataset,
            batch_size=config['batch_size'],
            sampler=train_sampler,
            num_workers=config['num_workers'],
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True if config['num_workers'] > 0 else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            sampler=val_sampler,
            num_workers=config['num_workers'],
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True if config['num_workers'] > 0 else False
        )
        
        # Create model
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
        
        # Create trainer
        trainer = MultiGPUTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            rank=rank,
            world_size=world_size
        )
        
        # Load checkpoint if exists
        if config.get('resume_checkpoint'):
            trainer.load_checkpoint(config['resume_checkpoint'])
        
        # Train
        trainer.train(num_epochs=config['epochs'])
        
    except Exception as e:
        print(f"❌ Error in rank {rank}: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        cleanup_distributed()


# =====================================================================
# MAIN FUNCTION
# =====================================================================
def main():
    """Main entry point"""
    
    # Configuration
    config = {
        # Paths (FIXED for Kaggle)
        'data_dir': '/kaggle/input/flickr8k',
        'vocab_dir': '/kaggle/input/vit-gpt2-nano/checkpoints',
        'checkpoint_dir': '/kaggle/working/checkpoints',
        'resume_checkpoint': '/kaggle/input/vit-gpt2-nano/checkpoints/last_checkpoint.pt',
        
        # Model
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
        'dropout': 0.1,
        'drop_path_rate': 0.2,
        'num_registers': 4,
        
        # Training (adjusted for 2 GPUs)
        'batch_size': 32,  # Per GPU: 32 x 2 = 64 total
        'epochs': 2,
        'lr': 1e-4,
        'weight_decay': 0.02,
        'grad_clip': 1.0,
        'grad_accum_steps': 5,
        'label_smoothing': 0.15,
        
        # Early stopping
        'patience': 5,
        'min_delta': 0.001,
        
        # Optimization
        'use_amp': True,
        
        # Data
        'num_workers': 2,  # Per GPU
    }
    
    # Check GPU availability
    if not torch.cuda.is_available():
        raise RuntimeError("❌ CUDA not available!")
    
    world_size = torch.cuda.device_count()
    
    if world_size < 2:
        raise RuntimeError(f"❌ Need 2 GPUs, but only {world_size} available!")
    
    print("\n" + "="*70)
    print("🚀 MULTI-GPU TRAINING SETUP")
    print("="*70)
    print(f"🖥️ Available GPUs: {world_size}")
    for i in range(world_size):
        print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
    print("="*70 + "\n")
    
    # Enable optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    
    # Spawn processes
    mp.spawn(
        train_worker,
        args=(world_size, config),
        nprocs=world_size,
        join=True
    )
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETED!")
    print("="*70)


if __name__ == '__main__':
    main()
