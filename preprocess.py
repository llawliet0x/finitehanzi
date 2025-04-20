import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import re

class HanziDataset(Dataset):
    def __init__(self, img_dir, svg_dir, transform=None, max_seq_len=100):
        self.img_dir = Path(img_dir)
        self.svg_dir = Path(svg_dir)
        self.transform = transform
        self.max_seq_len = max_seq_len
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
        
        # 構建詞彙表
        self.vocab = self._build_vocab()
        self.vocab_size = len(self.vocab)
        
    def _build_vocab(self):
        # 基本命令詞彙
        commands = ['M', 'L', 'Q', 'Z']
        
        # 創建詞彙表
        vocab = ['<pad>', '<sos>', '<eos>'] + commands
        
        # 添加坐標值到詞彙表
        for i in range(1025):  # 0-1024
            vocab.append(str(i))
            
        return {token: idx for idx, token in enumerate(vocab)}
    
    def _extract_path_from_svg(self, svg_path):
        tree = ET.parse(svg_path)
        root = tree.getroot()
        path = root.find('.//{http://www.w3.org/2000/svg}path')
        return path.get('d') if path is not None else ''
    
    def _parse_path_commands(self, path_string):
        # 使用正則表達式分割命令和坐標
        pattern = r'([A-Za-z])\s*([-\d\s.]*)'
        matches = re.findall(pattern, path_string)
        
        tokens = []
        for cmd, coords in matches:
            # 添加命令token
            tokens.append(cmd)
            
            # 處理坐標
            if coords.strip():
                coord_list = coords.strip().split()
                for coord in coord_list:
                    # 將坐標值轉換為整數並添加到token中
                    coord_int = int(float(coord))
                    tokens.append(str(coord_int))
        
        return tokens
    
    def _path_to_tokens(self, path_string):
        # 將path字符串轉換為token序列
        tokens = [self.vocab['<sos>']]
        commands = self._parse_path_commands(path_string)
        
        for cmd in commands:
            if cmd in self.vocab:
                tokens.append(self.vocab[cmd])
        
        tokens.append(self.vocab['<eos>'])
        
        # 填充序列到固定長度
        if len(tokens) < self.max_seq_len:
            tokens.extend([self.vocab['<pad>']] * (self.max_seq_len - len(tokens)))
        else:
            tokens = tokens[:self.max_seq_len]
            tokens[-1] = self.vocab['<eos>']
            
        return tokens
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = self.img_dir / img_name
        svg_path = self.svg_dir / img_name.replace('.png', '.svg')
        
        # 讀取並處理圖像
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).unsqueeze(0)  # 1*H*W
        
        # 讀取並處理SVG
        path_string = self._extract_path_from_svg(svg_path)
        tokens = self._path_to_tokens(path_string)
        
        return {
            'image': img,
            'tokens': torch.tensor(tokens),
            'path_string': path_string
        }

def create_dataloaders(img_dir, svg_dir, batch_size=32, max_seq_len=100):
    dataset = HanziDataset(img_dir, svg_dir, max_seq_len=max_seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader, dataset.vocab

if __name__ == '__main__':
    # 示例用法
    train_loader, vocab = create_dataloaders(
        'dataset/imgs_train',
        'dataset/svgs_train',
        max_seq_len=100
    )
    
    # 打印詞彙表大小
    print(f"詞彙表大小: {len(vocab)}")
    
    # 打印一個batch的數據
    batch = next(iter(train_loader))
    print(f"圖像形狀: {batch['image'].shape}")
    print(f"Token序列長度: {batch['tokens'].shape}")
    
    # 打印一些token示例
    print("\nToken示例：")
    for i, (token, idx) in enumerate(vocab.items()):
        if i < 10:  # 打印前10個token
            print(f"{token}: {idx}") 