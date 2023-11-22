import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size=21, hidden_size=64, num_layers=2, embedding_size=10, dropout=0.05, output_size=20):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.embeddings = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers)
        
    def forward(self, x):
        embedding = self.embeddings(x)
        embedding = self.dropout(embedding)
        output, (hn, cn) = self.lstm(embedding)
        return output, hn, cn
    
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(38, 21, kernel_size=6, stride=2), # 64 -> 30
            nn.BatchNorm1d(21),
            nn.Sigmoid(),
            nn.Conv1d(21, 10, kernel_size=6, stride=3), # 30 -> 9
            nn.BatchNorm1d(10),
            nn.Sigmoid(),
            nn.Conv1d(10, 1, kernel_size=9, stride=1), # 9 -> 1
            nn.BatchNorm1d(1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.model(x)
    
class peptide2RT(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, x):
        output, hn, cn = encoder(x)
        output = decoder(output)
        return output