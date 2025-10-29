"""
Training Script cho vitXgpt-nano trên Flickr8k Dataset - FIXED VERSION
Tối ưu cho Kaggle với 2x GPU T4 (16GB VRAM mỗi GPU)

FIXES:
- Thêm validation để detect token issues
- Sửa tokenization logic
- Thêm error handling cho data loading

Usage:
    Single GPU: CUDA_LAUNCH_BLOCKING=1 python train.py
    Multi GPU:  torchrun --nproc_per_node=2 train.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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

# Import model từ file model.py trong cùng thư mục
from model import ModernImageCaptioningModel


# =====================================================================
# XỬ LÝ DỮ LIỆU FLICKR8K
# =====================================================================
class Flickr8kProcessor:
    """
    Xử lý dataset Flickr8k từ Kaggle
    Cấu trúc dữ liệu Flickr8k:
    - Images/: thư mục chứa 8000 ảnh
    - captions.txt: file chứa 5 captions cho mỗi ảnh (40,000 captions)
    """
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / 'Images'
        self.captions_file = self.data_dir / 'captions.txt'
        
    def load_and_clean_captions(self) -> pd.DataFrame:
        """
        Load và làm sạch captions
        Format: image,caption
        """
        print("📖 Đang đọc captions...")
        df = pd.read_csv(self.captions_file)
        
        # Clean captions
        df['caption'] = df['caption'].str.lower()
        df['caption'] = df['caption'].str.replace('[^a-z0-9\\s]', '', regex=True)
        df['caption'] = df['caption'].str.strip()
        
        # ✅ FIX: Thêm special tokens - KHÔNG thêm ở đây, sẽ thêm khi tokenize
        # df['caption'] = '<start> ' + df['caption'] + ' <end>'
        
        print(f"✓ Loaded {len(df)} captions từ {df['image'].nunique()} ảnh")
        return df
    
    def build_vocabulary(self, df: pd.DataFrame, min_freq: int = 5) -> Dict:
        """
        Xây dựng vocabulary từ captions
        
        Args:
            df: DataFrame chứa captions
            min_freq: Tần suất xuất hiện tối thiểu của từ
        
        Returns:
            word2idx: Dict ánh xạ từ -> index
            idx2word: Dict ánh xạ index -> từ
        """
        print(f"🔤 Xây dựng vocabulary (min_freq={min_freq})...")
        
        # Đếm tần suất từ
        word_freq = {}
        for caption in df['caption']:
            for word in caption.split():
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Lọc từ theo tần suất
        vocab = [word for word, freq in word_freq.items() if freq >= min_freq]
        
        # Special tokens
        special_tokens = ['<pad>', '<start>', '<end>', '<unk>']
        vocab = special_tokens + sorted(vocab)
        
        word2idx = {word: idx for idx, word in enumerate(vocab)}
        idx2word = {idx: word for word, idx in word2idx.items()}
        
        print(f"✓ Vocabulary size: {len(vocab)} từ")
        print(f"  - Tổng từ unique: {len(word_freq)}")
        print(f"  - Từ xuất hiện >= {min_freq} lần: {len(vocab) - 4}")
        
        return word2idx, idx2word
    
    def split_dataset(self, df: pd.DataFrame, train_ratio: float = 0.8, 
                     val_ratio: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Chia dataset thành train/val/test
        
        Chiến lược: Chia theo ảnh, không theo caption để tránh data leakage
        """
        print(f"✂️ Chia dataset (train={train_ratio}, val={val_ratio})...")
        
        # Lấy danh sách ảnh unique
        unique_images = df['image'].unique()
        np.random.seed(42)
        np.random.shuffle(unique_images)
        
        # Tính số lượng ảnh cho mỗi split
        n_images = len(unique_images)
        n_train = int(n_images * train_ratio)
        n_val = int(n_images * val_ratio)
        
        train_images = unique_images[:n_train]
        val_images = unique_images[n_train:n_train + n_val]
        test_images = unique_images[n_train + n_val:]
        
        # Tách captions theo ảnh
        train_df = df[df['image'].isin(train_images)].reset_index(drop=True)
        val_df = df[df['image'].isin(val_images)].reset_index(drop=True)
        test_df = df[df['image'].isin(test_images)].reset_index(drop=True)
        
        print(f"✓ Train: {len(train_df)} captions từ {len(train_images)} ảnh")
        print(f"✓ Val: {len(val_df)} captions từ {len(val_images)} ảnh")
        print(f"✓ Test: {len(test_df)} captions từ {len(test_images)} ảnh")
        
        return train_df, val_df, test_df


# =====================================================================
# DATASET VÀ DATALOADER TỐI ƯU - FIXED VERSION
# =====================================================================
class Flickr8kDataset(Dataset):
    """
    Dataset tối ưu cho training với các đặc điểm:
    - Data augmentation mạnh cho ảnh
    - Efficient tokenization
    - Memory-friendly (không cache toàn bộ ảnh)
    
    ✅ FIXED: Proper tokenization và validation
    """
    
    def __init__(
        self, 
        df: pd.DataFrame,
        images_dir: str,
        word2idx: Dict,
        max_length: int = 128,
        img_size: int = 224,
        is_train: bool = True
    ):
        self.df = df
        self.images_dir = Path(images_dir)
        self.word2idx = word2idx
        self.max_length = max_length
        self.is_train = is_train
        
        # ✅ Cache special token indices
        self.pad_idx = word2idx['<pad>']
        self.start_idx = word2idx['<start>']
        self.end_idx = word2idx['<end>']
        self.unk_idx = word2idx['<unk>']
        self.vocab_size = len(word2idx)
        
        # Data augmentation cho training
        if is_train:
            self.transform = transforms.Compose([
                transforms.Resize((img_size + 32, img_size + 32), 
                                interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomCrop((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, 
                                     saturation=0.2, hue=0.1),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size), 
                                interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def tokenize(self, caption: str) -> List[int]:
        """
        ✅ FIXED: Tokenize caption thành list of indices
        Format: [<start>, word1, word2, ..., <end>, <pad>, ...]
        """
        words = caption.split()
        
        # Convert words to indices
        tokens = [self.word2idx.get(word, self.unk_idx) for word in words]
        
        # Add start and end tokens
        tokens = [self.start_idx] + tokens + [self.end_idx]
        
        # Truncate if too long (giữ lại start và end)
        if len(tokens) > self.max_length:
            tokens = [self.start_idx] + tokens[1:self.max_length-1] + [self.end_idx]
        
        # Pad to max_length
        if len(tokens) < self.max_length:
            tokens = tokens + [self.pad_idx] * (self.max_length - len(tokens))
        
        return tokens
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load và transform ảnh
        img_path = self.images_dir / row['image']
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            print(f"⚠️ Lỗi đọc ảnh {img_path}: {e}")
            # Return một ảnh trống nếu lỗi
            image = torch.zeros(3, 224, 224)
        
        # ✅ FIXED: Tokenize caption đúng cách
        tokens = self.tokenize(row['caption'])
        tokens = torch.tensor(tokens, dtype=torch.long)
        
        # ✅ VALIDATION: Đảm bảo tokens hợp lệ
        assert tokens.min() >= 0, f"Negative token: {tokens.min()}"
        assert tokens.max() < self.vocab_size, f"Token {tokens.max()} >= vocab_size {self.vocab_size}"
        
        # Tạo input và target cho teacher forcing
        # Input: [<start>, word1, word2, ..., wordN]
        # Target: [word1, word2, ..., wordN, <end>]
        input_ids = tokens[:-1]
        target_ids = tokens[1:]
        
        # Tạo attention mask (1 cho token thật, 0 cho padding)
        attention_mask = (target_ids != self.pad_idx).long()
        
        return {
            'image': image,
            'input_ids': input_ids,
            'target_ids': target_ids,
            'attention_mask': attention_mask,
            'caption': row['caption']
        }


# ✅ THÊM HÀM VALIDATION
def validate_dataloader(dataloader, vocab_size, num_batches=5):
    """
    Validate dataloader để đảm bảo không có lỗi với tokens
    """
    print("\n" + "="*70)
    print("🔍 VALIDATING DATALOADER")
    print("="*70)
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
        
        input_ids = batch['input_ids']
        target_ids = batch['target_ids']
        attention_mask = batch['attention_mask']
        
        print(f"\nBatch {batch_idx}:")
        print(f"  Input shape: {input_ids.shape}")
        print(f"  Target shape: {target_ids.shape}")
        print(f"  Input range: [{input_ids.min()}, {input_ids.max()}]")
        print(f"  Target range: [{target_ids.min()}, {target_ids.max()}]")
        print(f"  Vocab size: {vocab_size}")
        
        # Kiểm tra có token nào vượt vocab_size không
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
# TRAINER TỐI ƯU CHO 2 GPU T4 - FIXED VERSION
# =====================================================================
class Flickr8kTrainer:
    """
    Trainer được tối ưu cho Kaggle với 2x T4 GPU (16GB mỗi GPU)
    
    ✅ FIXED: Thêm validation và error handling
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
        """Tạo optimizer với weight decay riêng biệt cho các loại parameters"""
        decay = []
        no_decay = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # Không áp dụng weight decay cho bias, norm, và embedding
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
        """Tạo OneCycleLR scheduler với warmup"""
        total_steps = len(self.train_loader) * self.config.get('epochs', 20) // self.grad_accum_steps
        
        scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.config.get('lr', 3e-4),
            total_steps=total_steps,
            pct_start=0.05,  # 5% warmup
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=1e4
        )
        
        return scheduler
    
    def train_epoch(self):
        """Train một epoch"""
        self.model.train()
        total_loss = 0
        total_tokens = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch}', 
                   disable=not self.is_main)
        
        for step, batch in enumerate(pbar):
            try:
                # Move to device
                images = batch['image'].to(self.device, non_blocking=True)
                input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                target_ids = batch['target_ids'].to(self.device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
                
                # ✅ VALIDATION: Kiểm tra input trước khi forward
                if step == 0 and self.epoch == 0:
                    print(f"\n🔍 First batch validation:")
                    print(f"  Input range: [{input_ids.min()}, {input_ids.max()}]")
                    print(f"  Target range: [{target_ids.min()}, {target_ids.max()}]")
                    print(f"  Vocab size: {self.config['vocab_size']}")
                
                # Mixed precision forward
                with autocast(enabled=self.use_amp):
                    logits = self.model(images, input_ids)
                    
                    # ✅ VALIDATION: Kiểm tra output shape
                    if step == 0 and self.epoch == 0:
                        print(f"  Logits shape: {logits.shape}")
                        print(f"  Expected vocab dim: {self.config['vocab_size']}")
                    
                    # Compute loss với label smoothing
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        target_ids.reshape(-1),
                        ignore_index=self.config.get('pad_idx', 0),
                        label_smoothing=self.label_smoothing
                    )
                    
                    # Scale loss cho gradient accumulation
                    loss = loss / self.grad_accum_steps
                
                # Backward
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Update weights sau gradient accumulation
                if (step + 1) % self.grad_accum_steps == 0:
                    # Gradient clipping
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    
                    # Optimizer step
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
                
                # Update progress bar
                if self.is_main:
                    pbar.set_postfix({
                        'loss': f'{batch_loss:.4f}',
                        'ppl': f'{np.exp(batch_loss):.2f}',
                        'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                    })
            
            except Exception as e:
                print(f"\n❌ Error at step {step}:")
                print(f"  Error: {e}")
                print(f"  Input shape: {input_ids.shape if 'input_ids' in locals() else 'N/A'}")
                print(f"  Target shape: {target_ids.shape if 'target_ids' in locals() else 'N/A'}")
                raise
        
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
    
    def save_checkpoint(self, is_best: bool = False):
        """Save checkpoint"""
        if not self.is_main:
            return
        
        # Get model state (unwrap DDP)
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
            'config': self.config
        }
        
        # Save regular checkpoint
        ckpt_path = self.checkpoint_dir / f'checkpoint_epoch_{self.epoch}.pt'
        torch.save(checkpoint, ckpt_path)
        print(f"💾 Saved checkpoint: {ckpt_path}")
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"🏆 Saved best model: {best_path}")
    
    def train(self, num_epochs: int):
        """Full training loop"""
        print("\n" + "="*70)
        print("🚀 BẮT ĐẦU TRAINING")
        print("="*70)
        
        # Model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        if self.is_main:
            print(f"📊 Tổng parameters: {total_params/1e6:.2f}M")
            print(f"📊 Trainable parameters: {trainable_params/1e6:.2f}M")
            print(f"📊 Non-trainable parameters: {(total_params-trainable_params)/1e6:.2f}M")
            print(f"🔧 Gradient accumulation steps: {self.grad_accum_steps}")
            gpu_count = torch.cuda.device_count() if dist.is_initialized() else 1
            print(f"🔧 Effective batch size: {self.config['batch_size'] * self.grad_accum_steps * gpu_count}")
            print("="*70 + "\n")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Set epoch for distributed sampler
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
                print(f"{'='*70}\n")
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            self.save_checkpoint(is_best=is_best)
            
            # Save every N epochs
            if (epoch + 1) % self.config.get('save_every', 5) == 0:
                self.save_checkpoint()
        
        if self.is_main:
            print("\n" + "="*70)
            print("✅ HOÀN THÀNH TRAINING")
            print(f"🏆 Best validation loss: {self.best_val_loss:.4f}")
            print("="*70)


# =====================================================================
# MAIN FUNCTION - SETUP VÀ CHẠY TRAINING
# =====================================================================
def setup_distributed():
    """Setup distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        
        return local_rank, rank, world_size
    
    return 0, 0, 1


def main():
    """
    Main training script
    
    Cấu trúc dữ liệu Kaggle Flickr8k:
    /kaggle/input/flickr8k/
        ├── Images/
        │   ├── 1000268201_693b08cb0e.jpg
        │   └── ...
        └── captions.txt
    """
    
    # ===== CONFIGURATION =====
    config = {
        # Paths (sửa theo cấu trúc Kaggle của bạn)
        'data_dir': '/kaggle/input/flickr8k',
        'checkpoint_dir': '/kaggle/working/checkpoints',
        
        # Model architecture
        'vocab_size': None,  # Sẽ được set sau khi build vocabulary
        'img_size': 224,
        'patch_size': 16,
        'max_seq_len': 128,
        'embed_dim': 768,
        'encoder_depth': 12,
        'decoder_depth': 12,
        'num_heads': 12,
        'num_kv_heads': 4,  # GQA
        'mlp_ratio': 4.0,
        'dropout': 0.0,
        'drop_path_rate': 0.1,
        'num_registers': 4,
        
        # Training hyperparameters (tối ưu cho T4)
        'batch_size': 12,
        'epochs': 12,
        'lr': 3e-4,
        'weight_decay': 0.01,
        'grad_clip': 1.0,
        'grad_accum_steps': 4,
        'label_smoothing': 0.1,
        
        # Optimization
        'use_amp': True,
        
        # Data
        'num_workers': 4,
        'pin_memory': True,
        'prefetch_factor': 2,
        'min_word_freq': 2,
        
        # Logging
        'save_every': 5,
    }
    
    # Setup distributed
    local_rank, rank, world_size = setup_distributed()
    device = f'cuda:{local_rank}'
    is_main = (rank == 0)
    
    if is_main:
        print("\n" + "="*70)
        print("🎯 KHỞI TẠO TRAINING PIPELINE")
        print("="*70)
        print(f"🖥️ Device: {world_size}x {torch.cuda.get_device_name(local_rank)}")
        print(f"💾 VRAM: {torch.cuda.get_device_properties(local_rank).total_memory / 1e9:.1f} GB")
    
    # ===== STEP 1: PROCESS DATA =====
    if is_main:
        print("\n" + "-"*70)
        print("STEP 1: Xử lý Dataset Flickr8k")
        print("-"*70)
    
    processor = Flickr8kProcessor(config['data_dir'])
    
    # Load captions
    df = processor.load_and_clean_captions()
    
    # Build vocabulary
    word2idx, idx2word = processor.build_vocabulary(df, min_freq=config['min_word_freq'])
    config['vocab_size'] = len(word2idx)
    config['pad_idx'] = word2idx['<pad>']
    
    # Split dataset
    train_df, val_df, test_df = processor.split_dataset(df)
    
    # Save vocabulary
    if is_main:
        vocab_path = Path(config['checkpoint_dir']) / 'vocab.json'
        vocab_path.parent.mkdir(exist_ok=True, parents=True)
        with open(vocab_path, 'w') as f:
            json.dump({'word2idx': word2idx, 'idx2word': idx2word}, f)
        print(f"💾 Saved vocabulary to {vocab_path}")
    
    # ===== STEP 2: CREATE DATASETS =====
    if is_main:
        print("\n" + "-"*70)
        print("STEP 2: Tạo DataLoaders")
        print("-"*70)
    
    images_dir = processor.images_dir
    
    train_dataset = Flickr8kDataset(
        train_df,
        images_dir,
        word2idx,
        max_length=config['max_seq_len'],
        img_size=config['img_size'],
        is_train=True
    )
    
    val_dataset = Flickr8kDataset(
        val_df,
        images_dir,
        word2idx,
        max_length=config['max_seq_len'],
        img_size=config['img_size'],
        is_train=False
    )
    
    # Create samplers for distributed training
    train_sampler = DistributedSampler(
        train_dataset, 
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    ) if world_size > 1 else None
    
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    ) if world_size > 1 else None
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
        prefetch_factor=config['prefetch_factor'],
        persistent_workers=True,
        drop_last=True  # Quan trọng cho DDP
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        sampler=val_sampler,
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
        prefetch_factor=config['prefetch_factor'],
        persistent_workers=True
    )
    
    if is_main:
        print(f"✓ Train batches: {len(train_loader)}")
        print(f"✓ Val batches: {len(val_loader)}")
        print(f"✓ Effective batch size: {config['batch_size'] * world_size * config['grad_accum_steps']}")
    
    # ✅ VALIDATE DATALOADER TRƯỚC KHI TRAINING
    if is_main:
        validate_dataloader(train_loader, config['vocab_size'], num_batches=3)
    
    # ===== STEP 3: CREATE MODEL =====
    if is_main:
        print("\n" + "-"*70)
        print("STEP 3: Khởi tạo Model")
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
    
    # ===== STEP 4: CREATE TRAINER =====
    if is_main:
        print("\n" + "-"*70)
        print("STEP 4: Khởi tạo Trainer")
        print("-"*70)
    
    trainer = Flickr8kTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        local_rank=local_rank
    )
    
    # ===== STEP 5: TRAIN =====
    if is_main:
        print("\n" + "-"*70)
        print("STEP 5: Bắt đầu Training")
        print("-"*70)
    
    trainer.train(num_epochs=config['epochs'])
    
    # Cleanup
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    # ✅ ENABLE CUDA_LAUNCH_BLOCKING để debug dễ hơn
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'
    
    # Enable TF32 (T4 không hỗ trợ nhưng không sao)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Benchmark mode cho performance
    torch.backends.cudnn.benchmark = True
    
    # Set seed cho reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run training
    main()

