import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 使用ResNet18作為基礎網絡
        resnet = resnet18(pretrained=True)
        # 修改第一層卷積以適應單通道輸入
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 移除最後的全連接層
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
    def forward(self, x):
        # x: (B, 1, H, W)
        features = self.backbone(x)  # (B, 512, H/32, W/32)
        return features

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(d_model, vocab_size)
        
    def forward(self, memory, tgt, tgt_mask=None, tgt_key_padding_mask=None):
        # memory: (H*W, B, d_model)
        # tgt: (T, B)
        tgt = self.embedding(tgt) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float))
        tgt = self.pos_encoder(tgt)
        
        output = self.transformer_decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        output = self.fc(output)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: (T, B, d_model)
        x = x + self.pe[:, :x.size(0)]
        return x

class HanziModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(vocab_size)
        
    def forward(self, images, tokens, tgt_mask=None, tgt_key_padding_mask=None):
        # images: (B, 1, H, W)
        # tokens: (T, B)
        
        # 編碼器處理圖像
        features = self.encoder(images)  # (B, 512, H/32, W/32)
        B, C, H, W = features.shape
        features = features.view(B, C, -1).permute(2, 0, 1)  # (H*W, B, C)
        
        # 解碼器生成序列
        output = self.decoder(
            memory=features,
            tgt=tokens,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        return output
    
    def generate(self, images, max_len=100):
        self.eval()
        with torch.no_grad():
            # 編碼器處理圖像
            features = self.encoder(images)
            B, C, H, W = features.shape
            features = features.view(B, C, -1).permute(2, 0, 1)
            
            # 初始化輸出序列
            outputs = torch.ones(1, B, dtype=torch.long, device=images.device) * self.decoder.embedding.weight.shape[0] - 2  # <sos>
            
            for _ in range(max_len):
                output = self.decoder(features, outputs)
                next_token = output[-1:, :, :].argmax(dim=-1)
                outputs = torch.cat([outputs, next_token], dim=0)
                
                if (next_token == self.decoder.embedding.weight.shape[0] - 1).all():  # <eos>
                    break
            
            return outputs 