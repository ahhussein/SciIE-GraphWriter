from torch import nn
import torch
import torch.nn.functional as nnf
import util
class CharEmbeddings(nn.Module):
    def __init__(self, config, data):
        super().__init__()
        self.config = config
        self.data = data
        self.cnns = nn.ModuleList(nn.Conv1d(
            self.config['char_embedding_size'],
            config['filter_size'],
            filter_width
        ) for filter_width in self.config['filter_widths'])

        self.cnns.apply(self._init_weights)

        emb = torch.empty(data.dict_size, data.char_embedding_size)

        nn.init.xavier_uniform_(emb)
        self.embeddings = nn.Parameter(emb)

    def forward(self, char_index):
        # number-sentences x max-sentence-length x max-word-length (over all sentences)
        num_sentences = char_index.shape[0]
        max_sentence_length = char_index.shape[1]

        # [num_sentences, max_sentence_length, max_word_length, emb]
        char_emb = self.embeddings[char_index]

        # [num_sentences * max_sentence_length, max_word_length, emb]
        flattened_char_emb = char_emb.view(num_sentences * max_sentence_length, char_emb.shape[2], -1)

        flattened_char_emb = torch.transpose(flattened_char_emb, 2, 1)
        outputs = []
        for cnn in self.cnns:
            conv, indices = torch.max(nnf.relu(cnn(flattened_char_emb)), 2)
            outputs.append(conv)
        concatenated = torch.cat(outputs, 1) #[num-words, num-filters * len(filter-sizes)]

        return concatenated.view(num_sentences, max_sentence_length, -1) # [num_sentences, max_sentence_length, emb]

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            torch.nn.init.uniform_(m.bias, -1 * util.golort_factor(m.bias.shape[0]), util.golort_factor(m.bias.shape[0]))


class ElmoEmbeddings(nn.Module):
    def __init__(self, config, data):
        super().__init__()
        self.data = data
        weight = torch.empty(data.lm_layers)
        nn.init.constant_(weight, 0.0)
        self.weights = nn.Parameter(weight)

        scalar = torch.empty(1)
        nn.init.constant_(scalar, 1.0)
        self.scalar = nn.Parameter(scalar)
        self.softmax = nn.Softmax(0)

    def forward(self, lm_emb):
        num_sentences = lm_emb.shape[0]
        max_sentence_length = lm_emb.shape[1]
        emb_size = lm_emb.shape[2]
        # [num_sentences * max_sentence_length * emb, layers]
        flattened_lm_emb = lm_emb.view(num_sentences * max_sentence_length * emb_size, self.data.lm_layers)

        # [num_sentences * max_sentence_length * emb, 1]
        flattened_aggregated_lm_emb = torch.matmul(flattened_lm_emb, self.softmax(self.weights).unsqueeze(1))
        # [num_sentences, max_sentence_length, emb]
        return (
            flattened_aggregated_lm_emb.view(num_sentences, max_sentence_length, emb_size) * self.scalar,
            self.weights,
            self.scalar
        )


class Embeddings(nn.Module):
    def __init__(self, config, data):
        super().__init__()
        self.data = data
        self.config = config
        self.char_embeddings = CharEmbeddings(config, data)
        self.elmo_embeddings = ElmoEmbeddings(config, data)
        self.dropout = self.config['lexical_dropout_rate']

    def forward(self, batch):
        # [num_sentences, max_sentence_length, emb-context]
        context_emb_list = [batch.context_word_emb]
        head_emb_list = [batch.head_word_emb]

        # calculate and append char embeddings
        # [num_sentences, max_sentence_length, emb-char]
        aggregated_char_emb = self.char_embeddings(batch.char_idx)
        context_emb_list.append(aggregated_char_emb)
        head_emb_list.append(aggregated_char_emb)

        # calculate and append elmo embeddings
        # [num_sentences, max_sentence_length, emb-lm]
        aggregated_lm_emb, lm_weights, lm_scaling = self.elmo_embeddings(batch.lm_emb)
        context_emb_list.append(aggregated_lm_emb)

        # Concatenate and apply dropout
        context_emb = torch.cat(context_emb_list, 2) # [num_sentences, max_sentence_length, emb1]
        head_emb = torch.cat(head_emb_list, 2) # [num_sentences, max_sentence_length, emb2]

        context_emb = nnf.dropout(context_emb, self.dropout)
        head_emb = nnf.dropout(head_emb, self.dropout)

        return context_emb, head_emb, lm_weights, lm_scaling



