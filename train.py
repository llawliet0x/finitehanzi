import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from preprocess import HanziDataset, create_dataloaders
from model import HanziModel
import os
from tqdm import tqdm

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(train_loader, desc='Training'):
        images = batch['image'].to(device)
        tokens = batch['tokens'].to(device)
        
        # 創建目標掩碼
        tgt_mask = model.decoder.transformer_decoder.generate_square_subsequent_mask(tokens.size(0)).to(device)
        
        # 前向傳播
        output = model(images, tokens[:-1], tgt_mask=tgt_mask)
        
        # 計算損失
        loss = criterion(output.view(-1, output.size(-1)), tokens[1:].view(-1))
        
        # 反向傳播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validating'):
            images = batch['image'].to(device)
            tokens = batch['tokens'].to(device)
            
            tgt_mask = model.decoder.transformer_decoder.generate_square_subsequent_mask(tokens.size(0)).to(device)
            
            output = model(images, tokens[:-1], tgt_mask=tgt_mask)
            loss = criterion(output.view(-1, output.size(-1)), tokens[1:].view(-1))
            
            total_loss += loss.item()
    
    return total_loss / len(val_loader)

def main():
    # 設置設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 創建數據加載器
    train_loader, vocab = create_dataloaders(
        'dataset/imgs_train',
        'dataset/svgs_train',
        batch_size=32
    )
    
    val_loader, _ = create_dataloaders(
        'dataset/imgs_test',
        'dataset/svgs_test',
        batch_size=32
    )
    
    # 創建模型
    model = HanziModel(len(vocab)).to(device)
    
    # 定義損失函數和優化器
    criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # 訓練參數
    num_epochs = 100
    best_val_loss = float('inf')
    
    # 創建保存目錄
    os.makedirs('checkpoints', exist_ok=True)
    
    # 訓練循環
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        
        # 訓練
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f'Training Loss: {train_loss:.4f}')
        
        # 驗證
        val_loss = validate(model, val_loader, criterion, device)
        print(f'Validation Loss: {val_loss:.4f}')
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'vocab': vocab
            }, 'checkpoints/best_model.pth')
        
        # 保存檢查點
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'vocab': vocab
            }, f'checkpoints/checkpoint_epoch_{epoch+1}.pth')

if __name__ == '__main__':
    main() 