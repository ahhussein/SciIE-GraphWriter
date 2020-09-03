from torch import nn
import torch

class LSTMContextualize(nn.Module):
    def __init__(self, config, data):
        super().__init__()
        self.config = config
        self.data = data

        self.lstms = nn.ModuleList(nn.LSTM(
                        config['embedding_size_contactenated'],
                        config['contextualization_size'],
                        bidirectional=True,
                        num_layers=1,
                        batch_first=True
                    ) for i in self.config['contextualization_layers'])

        self.dropout = nn.Dropout(1 - self.config['lstm_dropout_rate'])

        self.ffnn = nn.Linear(
            config['contextualization_size'] * 2,
            config['contextualization_size'] * 2
        )

        torch.nn.init.xavier_uniform_(self.ffnn.weight)


    def forward(self, context_emb):
        current_inputs = context_emb
        for i, lstm in enumerate(self.lstms):
            e, (h, c) = self.lstm(current_inputs)
            e = self.dropout(e)


            if i > 0:
                # TODO debug
                highway_gates = torch.sigmoid(self.ffnn(e))

                e = highway_gates * e + (1 - highway_gates) * current_inputs

            current_inputs = e

            # TODO custom LSTM cell
            # [max-sent-length, num-sentences, num-dir=2 * hz]
        return e
