import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(input_size, output_size, kernel_size, stride),
            nn.BatchNorm1d(output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
        
class Encoder(nn.Module):
    def __init__(self, input_size=21, hidden_size=128, num_layers=2, embedding_size=10, dropout=0.3):
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
            ConvBlock(38, 24, 6, 2),
            ConvBlock(24, 12, 6, 5),
            ConvBlock(12, 1, 12, 1)
        )
        
    def forward(self, x):        
        return self.model(x)
    
class peptide2RT(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, x):
        output, hn, cn = self.encoder(x)
        output = self.decoder(output)
        return output