import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class HanziDataset(Dataset):
    def __init__(self, img_dir, svg_dir, transform=None):
        self.img_dir = Path(img_dir)
        self.svg_dir = Path(svg_dir)
        self.transform = transform
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
        
        # 構建詞彙表
        self.vocab = self._build_vocab()
        self.vocab_size = len(self.vocab)
        
    def _build_vocab(self):
        # 收集所有SVG文件中的path命令
        commands = set()
        for svg_file in os.listdir(self.svg_dir):
            if svg_file.endswith('.svg'):
                path = self._extract_path_from_svg(self.svg_dir / svg_file)
                commands.update(self._parse_path_commands(path))
        
        # 創建詞彙表
        vocab = ['<pad>', '<sos>', '<eos>'] + sorted(list(commands))
        return {token: idx for idx, token in enumerate(vocab)}
    
    def _extract_path_from_svg(self, svg_path):
        tree = ET.parse(svg_path)
        root = tree.getroot()
        # 找到path元素
        path = root.find('.//{http://www.w3.org/2000/svg}path')
        return path.get('d') if path is not None else ''
    
    def _parse_path_commands(self, path_string):
        # 解析SVG path命令
        commands = []
        current_command = ''
        for char in path_string:
            if char.isalpha():
                if current_command:
                    commands.append(current_command)
                current_command = char
            else:
                current_command += char
        if current_command:
            commands.append(current_command)
        return commands
    
    def _path_to_tokens(self, path_string):
        # 將path字符串轉換為token序列
        commands = self._parse_path_commands(path_string)
        tokens = [self.vocab['<sos>']]
        for cmd in commands:
            if cmd in self.vocab:
                tokens.append(self.vocab[cmd])
        tokens.append(self.vocab['<eos>'])
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

def create_dataloaders(img_dir, svg_dir, batch_size=32):
    dataset = HanziDataset(img_dir, svg_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader, dataset.vocab

if __name__ == '__main__':
    # 示例用法
    train_loader, vocab = create_dataloaders(
        'dataset/imgs_train',
        'dataset/svgs_train'
    )
    
    # 打印詞彙表大小
    print(f"詞彙表大小: {len(vocab)}")
    
    # 打印一個batch的數據
    batch = next(iter(train_loader))
    print(f"圖像形狀: {batch['image'].shape}")
    print(f"Token序列長度: {batch['tokens'].shape}") 