from torch import nn
import torch

class LSTMContextualize(nn.Module):
    def __init__(self, config, data, is_training=1):
        super().__init__()
        self.config = config
        self.data = data

        self.lstm = nn.LSTM(
                        config['embedding_size_contactenated'],
                        config['contextualization_size'],
                        bidirectional=True,
                        num_layers=config['contextualization_layers'],
                        batch_first=True
                    )

        self.dropout = nn.Dropout(1 - is_training * self.config['lstm_dropout_rate'])


    def forward(self, context_emb):
        e, (h,c) = self.lstm(context_emb)
        x = self.dropout(e)

        return x
