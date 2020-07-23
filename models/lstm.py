from torch import nn

class LSTMContextualize(nn.Module):
    def __init__(self, config, data):
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

        self.dropout = nn.Dropout(1 - self.config['lstm_dropout_rate'])


    def forward(self, context_emb):
        e, (h,c) = self.lstm(context_emb)
        # TODO custom LSTM cell and projection
        # [max-sent-length, num-sentences, num-dir=2 * hz]
        return self.dropout(e)
