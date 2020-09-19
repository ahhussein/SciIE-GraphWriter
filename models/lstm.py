from torch import nn
import torch
from models.lstm_custom import CustomLSTMCell, BidirLSTMLayer, LSTMState


class LSTMContextualize(nn.Module):
    def __init__(self, config, data):
        super().__init__()
        self.config = config
        self.data = data

        self.lstm = BidirLSTMLayer(
            CustomLSTMCell,
            self.config['lstm_dropout_rate'],
            config['embedding_size_contactenated'],
            config['contextualization_size']
        )

        self.dropout = nn.Dropout(self.config['lstm_dropout_rate'])

        states = [torch.empty(1, self.config['contextualization_size']) for i in range(4)]

        [torch.nn.init.xavier_uniform_(state) for state in states]

        self.states = nn.ParameterList([nn.Parameter(i) for i in states])

        # self.ffnn = nn.Linear(
        #     config['contextualization_size'] * 2,
        #     config['contextualization_size'] * 2
        # )
        #
        # torch.nn.init.xavier_uniform_(self.ffnn.weight)


    def forward(self, context_emb):
        #current_inputs = context_emb
        #for i, lstm in enumerate(self.lstms):
        context_emb = context_emb.transpose(0,1)

        states = [LSTMState(
            self.states[i].repeat(context_emb.shape[1], 1),
            self.states[2+i].repeat(context_emb.shape[1], 1)
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