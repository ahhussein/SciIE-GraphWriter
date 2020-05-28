from torch import nn, dtype
import torch
import util

class AntecedentScore(nn.Module):
    def __init__(self, config, is_training=1):
        super().__init__()
        self.config = config
        self.antecedent_distance_emb = nn.init.xavier_uniform_(
            torch.empty(10, self.config["feature_size"])
        )

        self.input = nn.Linear(
            3830, #TODO
            self.config["ffnn_size"]
        )

        self.hidden1 = nn.Linear(
            self.config["ffnn_size"],
            self.config["ffnn_size"]
        )

        self.output = nn.Linear(
            self.config["ffnn_size"],
            1
        )

        self.dropout = nn.Dropout(1 - is_training * self.config['dropout_rate'])
        torch.nn.init.xavier_uniform_(self.input.weight)
        torch.nn.init.xavier_uniform_(self.hidden1.weight)
        torch.nn.init.xavier_uniform_(self.output.weight)


    def forward(self, top_span_emb, top_span_mention_scores, antecedents):
        k = top_span_emb.shape[0]
        max_antecedents = antecedents.shape[1]
        feature_emb_list = []

        if self.config["use_features"]:
            target_indices = torch.arange(k)  # [k]
            antecedent_distance = target_indices.unsqueeze(1) - antecedents # [k, max_ant]
            antecedent_distance_buckets = util.bucket_distance(antecedent_distance)  # [k, max_ant]

            # TODO variable reuse with tf.variable_scope("features", reuse=reuse):
            antecedent_distance_emb = self.antecedent_distance_emb[antecedent_distance_buckets] # [k, max_ant]
            feature_emb_list.append(antecedent_distance_emb)

            feature_emb = self.dropout(torch.cat(feature_emb_list, 2)) # [k, max_ant, emb]

            antecedent_emb = top_span_emb[antecedents]  # [k, max_ant, emb]
            target_emb = top_span_emb.unsqueeze(1)  # [k, 1, emb]
            similarity_emb = antecedent_emb * target_emb  # [k, max_ant, emb]
            target_emb = target_emb.repeat([1, max_antecedents, 1])  # [k, max_ant, emb]
            pair_emb = torch.cat([target_emb, antecedent_emb, similarity_emb, feature_emb], 2)  # [k, max_ant, emb]

            #with tf.variable_scope("antecedent_scores", reuse=reuse):
            x = self.output(
                self.hidden1(
                    self.input(pair_emb)
                )
            )

            antecedent_scores = self.dropout(x) # [k, max_ant, 1]
            antecedent_scores = torch.squeeze(antecedent_scores, 2)  # [k, max_ant]
            # [k, max_ant]
            antecedent_scores += top_span_mention_scores.unsqueeze(1) + top_span_mention_scores[antecedents]
            return antecedent_scores, antecedent_emb, pair_emb  # [k, max_ant]








