from torch import nn
import torch
from models.lstm_custom import CustomLSTMCell, BidirLSTMLayer, LSTMState


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
                    ) for i in range(self.config['contextualization_layers']))

        self.lstm = BidirLSTMLayer(
            CustomLSTMCell,
            config['embedding_size_contactenated'],
            config['contextualization_size'],
            1 - self.config['lstm_dropout_rate']
        )

        self.dropout = nn.Dropout(1 - self.config['lstm_dropout_rate'])

        # self.ffnn = nn.Linear(
        #     config['contextualization_size'] * 2,
        #     config['contextualization_size'] * 2
        # )
        #
        # torch.nn.init.xavier_uniform_(self.ffnn.weight)


    def forward(self, context_emb, state = None):
        #current_inputs = context_emb
        #for i, lstm in enumerate(self.lstms):
        context_emb = context_emb.transpose(0,1)

        if not state:
            # TODO initilazation
            states = [torch.empty(
                    context_emb.shape[1],
                    self.config['contextualization_size']
                ) for i in range(4)]

            [torch.nn.init.xavier_uniform_(state) for state in states]

            states = [LSTMState(
                states[i],
                states[2+i]
            ) for i in range(2)]

        out, out_state = self.lstm(context_emb, states)

        e = self.dropout(out)


        # if i > 0:
        #     highway_gates = torch.sigmoid(self.ffnn(e))
        #
        #     e = highway_gates * e + (1 - highway_gates) * current_inputs

        # current_inputs = e

        # [max-sent-length, num-sentences, num-dir=2 * hz]
        return e.transpose(1,0)