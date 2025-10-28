"""
Inference Script cho vitXgpt-nano
Generate captions cho ảnh từ trained model

Usage:
    python inference.py --image path/to/image.jpg --checkpoint path/to/best_model.pt
    python inference.py --image path/to/image.jpg --checkpoint path/to/best_model.pt --method beam --beam_size 5
"""

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from pathlib import Path
import argparse
import json
from typing import Dict, List

from model import ModernImageCaptioningModel


class CaptionGenerator:
    """
    Generate captions từ trained model
    Hỗ trợ các sampling strategies: greedy, beam search
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        word2idx: Dict,
        idx2word: Dict,
        device: str = 'cuda',
        max_length: int = 50
    ):
        self.model = model
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.device = device
        self.max_length = max_length
        
        self.model.eval()
        self.model.to(device)
        
        # Special token IDs
        self.start_id = word2idx['<start>']
        self.end_id = word2idx['<end>']
        self.pad_id = word2idx['<pad>']
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), 
                            interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    @torch.no_grad()
    def generate_greedy(self, image: torch.Tensor) -> str:
        """
        Greedy decoding: chọn token có probability cao nhất ở mỗi bước
        Nhanh nhưng không đảm bảo caption tốt nhất
        """
        image = image.unsqueeze(0).to(self.device)
        
        # Start với <start> token
        input_ids = torch.tensor([[self.start_id]], device=self.device)
        
        for _ in range(self.max_length):
            # Forward pass
            logits = self.model(image, input_ids)
            
            # Lấy token có probability cao nhất
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            
            # Thêm vào sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Stop nếu gặp <end>
            if next_token.item() == self.end_id:
                break
        
        # Decode thành text
        token_ids = input_ids[0].tolist()
        caption = self._decode_tokens(token_ids)
        
        return caption
    
    @torch.no_grad()
    def generate_beam_search(
        self, 
        image: torch.Tensor, 
        beam_size: int = 5,
        length_penalty: float = 0.6
    ) -> str:
        """
        Beam search: duy trì beam_size sequences tốt nhất
        Chậm hơn greedy nhưng cho kết quả tốt hơn
        
        Args:
            image: Input image tensor
            beam_size: Số lượng beams
            length_penalty: Penalty cho caption dài (< 1 ưu tiên ngắn, > 1 ưu tiên dài)
        """
        image = image.unsqueeze(0).to(self.device)
        
        # Initialize beams
        beams = [(torch.tensor([[self.start_id]], device=self.device), 0.0)]  # (sequence, score)
        
        for step in range(self.max_length):
            new_beams = []
            
            for seq, score in beams:
                # Skip nếu sequence đã kết thúc
                if seq[0, -1].item() == self.end_id:
                    new_beams.append((seq, score))
                    continue
                
                # Forward pass
                logits = self.model(image, seq)
                log_probs = F.log_softmax(logits[:, -1, :], dim=-1)
                
                # Lấy top-k tokens
                top_log_probs, top_indices = log_probs.topk(beam_size, dim=-1)
                
                # Tạo new beams
                for i in range(beam_size):
                    new_token = top_indices[0, i].unsqueeze(0).unsqueeze(0)
                    new_seq = torch.cat([seq, new_token], dim=1)
                    new_score = score + top_log_probs[0, i].item()
                    
                    new_beams.append((new_seq, new_score))
            
            # Chọn top beam_size sequences
            # Apply length penalty
            scored_beams = [
                (seq, score / (len(seq[0]) ** length_penalty)) 
                for seq, score in new_beams
            ]
            beams = sorted(scored_beams, key=lambda x: x[1], reverse=True)[:beam_size]
            
            # Stop nếu tất cả beams đã kết thúc
            if all(seq[0, -1].item() == self.end_id for seq, _ in beams):
                break
        
        # Lấy best beam
        best_seq = beams[0][0]
        token_ids = best_seq[0].tolist()
        caption = self._decode_tokens(token_ids)
        
        return caption
    
    def _decode_tokens(self, token_ids: List[int]) -> str:
        """Convert token IDs thành text"""
        words = []
        for token_id in token_ids:
            if token_id == self.start_id:
                continue
            if token_id == self.end_id:
                break
            if token_id == self.pad_id:
                continue
            
            word = self.idx2word.get(str(token_id), '<unk>')
            words.append(word)
        
        return ' '.join(words)
    
    def generate_from_path(self, image_path: str, method: str = 'beam', **kwargs) -> str:
        """
        Generate caption từ image path
        
        Args:
            image_path: Đường dẫn đến ảnh
            method: 'greedy' hoặc 'beam'
            **kwargs: Additional arguments cho generation method
        """
        # Load và transform image
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        # Generate caption
        if method == 'greedy':
            caption = self.generate_greedy(image)
        elif method == 'beam':
            beam_size = kwargs.get('beam_size', 5)
            length_penalty = kwargs.get('length_penalty', 0.6)
            caption = self.generate_beam_search(image, beam_size, length_penalty)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return caption


def load_checkpoint(checkpoint_path: str, device: str = 'cuda'):
    """Load model từ checkpoint"""
    print(f"📦 Loading checkpoint từ {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
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
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✓ Loaded model from epoch {checkpoint['epoch']}")
    print(f"✓ Best validation loss: {checkpoint['best_val_loss']:.4f}")
    
    return model, config


def main():
    parser = argparse.ArgumentParser(description='Generate captions cho ảnh')
    parser.add_argument('--image', type=str, required=True, 
                       help='Đường dẫn đến ảnh')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Đường dẫn đến checkpoint')
    parser.add_argument('--vocab', type=str, default=None,
                       help='Đường dẫn đến vocab.json (default: cùng folder với checkpoint)')
    parser.add_argument('--method', type=str, default='beam',
                       choices=['greedy', 'beam'],
                       help='Generation method')
    parser.add_argument('--beam_size', type=int, default=5,
                       help='Beam size cho beam search')
    parser.add_argument('--length_penalty', type=float, default=0.6,
                       help='Length penalty cho beam search')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda hoặc cpu)')
    
    args = parser.parse_args()
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️ CUDA không available, chuyển sang CPU")
        args.device = 'cpu'
    
    # Load vocabulary
    if args.vocab is None:
        vocab_path = Path(args.checkpoint).parent / 'vocab.json'
    else:
        vocab_path = Path(args.vocab)
    
    print(f"📖 Loading vocabulary từ {vocab_path}...")
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    
    word2idx = vocab['word2idx']
    idx2word = vocab['idx2word']
    
    # Load model
    model, config = load_checkpoint(args.checkpoint, args.device)
    
    # Create generator
    generator = CaptionGenerator(
        model=model,
        word2idx=word2idx,
        idx2word=idx2word,
        device=args.device,
        max_length=config['max_seq_len']
    )
    
    # Generate caption
    print(f"\n🖼️ Đang generate caption cho: {args.image}")
    print(f"🔧 Method: {args.method}")
    if args.method == 'beam':
        print(f"🔧 Beam size: {args.beam_size}")
        print(f"🔧 Length penalty: {args.length_penalty}")
    
    caption = generator.generate_from_path(
        args.image,
        method=args.method,
        beam_size=args.beam_size,
        length_penalty=args.length_penalty
    )
    
    print(f"\n✨ Caption: {caption}\n")


if __name__ == '__main__':
    main()
