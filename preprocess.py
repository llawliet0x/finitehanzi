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
        # 特殊token
        special_tokens = ['<pad>', '<sos>', '<eos>']
        
        # 命令token
        command_tokens = ['M', 'Q', 'L', 'Z']
        
        # 坐標token (0-1024)
        coord_tokens = [str(i) for i in range(1025)]
        
        # 組合詞彙表
        vocab = special_tokens + command_tokens + coord_tokens
        return {token: idx for idx, token in enumerate(vocab)}
    
    def _extract_path_from_svg(self, svg_path):
        tree = ET.parse(svg_path)
        root = tree.getroot()
        path = root.find('.//{http://www.w3.org/2000/svg}path')
        return path.get('d') if path is not None else ''
    
    def _parse_path_commands(self, path_string):
        # 使用正則表達式匹配命令和坐標
        pattern = r'([A-Z])\s*([-\d\s.]*)'
        matches = re.findall(pattern, path_string)
        
        tokens = []
        for cmd, coords in matches:
            # 添加命令token
            tokens.append(cmd)
            
            # 根據不同命令處理坐標
            if cmd == 'M' or cmd == 'L':
                # M和L後面跟兩個坐標
                coords = coords.strip().split()
                if len(coords) >= 2:
                    tokens.extend([str(int(float(coords[0]))), str(int(float(coords[1])))])
            elif cmd == 'Q':
                # Q後面跟四個坐標
                coords = coords.strip().split()
                if len(coords) >= 4:
                    tokens.extend([
                        str(int(float(coords[0]))),  # 控制點x
                        str(int(float(coords[1]))),  # 控制點y
                        str(int(float(coords[2]))),  # 終點x
                        str(int(float(coords[3])))   # 終點y
                    ])
            # Z命令不需要坐標
        
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