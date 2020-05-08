from torch import nn

class Model(nn.Module):
    def __init__(self, config, is_training):
        super().__init__()
        self.config = config
        self.input_drop = nn.Dropout(1 - is_training * self.config["dropout_rate"])
        self.lexical_dropout = nn.Dropout(1 - is_training * self.config["lexical_dropout_rate"])
        self.lstm_dropout = nn.Dropout(1 - is_training * self.config["lstm_dropout_rate"])





