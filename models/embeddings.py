from torch import nn
import torch

class CharEmbeddings(nn.Module):
    def __init__(self, config, data):
        super().__init__()
        self.config = config
        self.data = data
        self.cnns= []
        for filter_width in self.config['filter_widths']:
            self.cnns.append(
                nn.Conv1d(
                    self.config['char_embedding_size'],
                    config['filter_size'],
                    filter_width
                )
            )

        self.relu = nn.ReLU()  # [num_words, num_chars - filter_size, num_filters]
        self.embeddings = nn.init.xavier_uniform_(torch.empty(len(data.char_dict), data.char_embedding_size))

    def forward(self, char_index):
        # number-sentences x max-sentence-length x max-word-length (over all sentences)
        num_sentences = char_index.shape[0]
        max_sentence_length = char_index.shape[1]

        char_emb = self.embeddings[char_index]
        flattened_char_emb = char_emb.view(num_sentences * max_sentence_length, char_emb.shape[2], -1)
        flattened_char_emb = torch.transpose(flattened_char_emb, 2, 1)

        outputs = []
        for cnn in self.cnns:
            conv, indices = torch.max(self.relu(cnn(flattened_char_emb)), 2)
            outputs.append(conv)

        concatenated = torch.cat(outputs, 1) #[num-words, num-filters * len(filter-sizes)]

        return concatenated.view(num_sentences, max_sentence_length, -1) # [num_sentences, max_sentence_length, emb]


class ElmoEmbeddings(nn.Module):
    def __init__(self, config, data):
        super().__init__()
        self.data = data
        self.weights = nn.init.constant_(torch.empty(data.lm_layers), 0.0)
        self.scalar = nn.init.constant_(torch.empty(1), 1.0)
        self.softmax = nn.Softmax(0)

    def forward(self, lm_emb):
        num_sentences = lm_emb.shape[0]
        max_sentence_length = lm_emb.shape[1]
        emb_size = lm_emb.shape[2]

        # [num_sentences * max_sentence_length * emb, layers]
        flattened_lm_emb = lm_emb.view(num_sentences * max_sentence_length * emb_size, self.data.lm_layers)

        # [num_sentences * max_sentence_length * emb, 1]
        flattened_aggregated_lm_emb = torch.matmul(flattened_lm_emb, self.softmax(self.weights).unsqueeze(1))

        # [num_sentences, max_sentence_length, emb
        return (
            flattened_aggregated_lm_emb.view(num_sentences, max_sentence_length, emb_size) * self.scalar,
            self.weights,
            self.scalar
        )


class Embeddings(nn.Module):
    def __init__(self, config, data, is_training=1):
        super().__init__()
        self.data = data
        self.config = config
        self.char_embeddings = CharEmbeddings(config, data)
        self.elmo_embeddings = ElmoEmbeddings(config, data)
        self.dropout = nn.Dropout(1 - is_training * self.config['lexical_dropout_rate'])

    def forward(self, batch):
        context_emb_list = [batch.context_word_emb]
        head_emb_list = [batch.head_word_emb]

        # calculate and append char embeddings
        aggregated_char_emb = self.char_embeddings(batch.char_idx)
        context_emb_list.append(aggregated_char_emb)
        head_emb_list.append(aggregated_char_emb)

        # calculate and append elmo embeddings
        aggregated_lm_emb, lm_weights, lm_scaling = self.elmo_embeddings(batch.lm_emb)
        context_emb_list.append(aggregated_lm_emb)

        # Concatenate and apply dropout
        context_emb = torch.cat(context_emb_list, 2) # [num_sentences, max_sentence_length, emb1]
        head_emb = torch.cat(head_emb_list, 2) # [num_sentences, max_sentence_length, emb2]

        context_emb = self.dropout(context_emb)
        head_emb = self.dropout(head_emb)

        return context_emb, head_emb, lm_weights, lm_scaling


