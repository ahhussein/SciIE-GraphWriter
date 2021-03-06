from torch import nn, dtype
import torch
import util
import torch.nn.functional as nnf

class AntecedentScore(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        emb = torch.empty(10, self.config["feature_size"])
        nn.init.xavier_uniform_(emb)
        self.antecedent_distance_emb = nn.Parameter(emb)

        self.input = nn.Linear(
            1520, #TODO
            self.config["ffnn_size"]
        )

        self.hidden = nn.Linear(
            self.config["ffnn_size"],
            self.config["ffnn_size"]
        )

        self.output = nn.Linear(
            self.config["ffnn_size"],
            1
        )

        self.dropout = self.config['dropout_rate']
        torch.nn.init.xavier_uniform_(self.input.weight)
        torch.nn.init.xavier_uniform_(self.hidden.weight)
        torch.nn.init.xavier_uniform_(self.output.weight)
        torch.nn.init.uniform_(self.input.bias, -1 * util.golort_factor(self.input.bias.shape[0]), util.golort_factor(self.input.bias.shape[0]))
        torch.nn.init.uniform_(self.hidden.bias, -1 * util.golort_factor(self.hidden.bias.shape[0]), util.golort_factor(self.hidden.bias.shape[0]))
        torch.nn.init.uniform_(self.output.bias, -1 * util.golort_factor(self.output.bias.shape[0]), util.golort_factor(self.output.bias.shape[0]))


    def forward(self, top_span_emb, top_span_mention_scores, antecedents):
        k = top_span_emb.shape[0]
        max_antecedents = antecedents.shape[1]
        feature_emb_list = []

        if self.config["use_features"]:
            antecedent_distance = torch.arange(k).unsqueeze(1) - antecedents # [k, max_ant]

            antecedent_distance_buckets = util.bucket_distance(antecedent_distance)  # [k, max_ant]

            antecedent_distance_emb = self.antecedent_distance_emb[antecedent_distance_buckets] # [k, max_ant, emb]
            feature_emb_list.append(antecedent_distance_emb)

        feature_emb = nnf.dropout(torch.cat(feature_emb_list, 2), self.dropout) # [k, max_ant, emb]


        # mentions are included in the mention_scores array since that array gives a score
        # of the possibility of having that span referenced as a mention
        antecedent_emb = top_span_emb[antecedents]  # [k, max_ant, emb]
        target_emb = top_span_emb.unsqueeze(1)  # [k, 1, emb]
        # all antecdents to be multiplied by the proform
        similarity_emb = antecedent_emb * target_emb  # [k, max_ant, emb]
        target_emb = target_emb.repeat([1, max_antecedents, 1])  # [k, max_ant, emb]
        pair_emb = torch.cat([target_emb, antecedent_emb, similarity_emb, feature_emb], 2)  # [k, max_ant, emb]

        #with tf.variable_scope("antecedent_scores", reuse=reuse):
        # candidate_span_emb = [num-candidates, emb]
        hidden1 = nnf.dropout(
            nnf.relu(
                self.input(pair_emb)
            ), self.dropout
        )

        hidden2 = nnf.dropout(
            nnf.relu(
                self.hidden(hidden1)
            ), self.dropout
        )

        antecedent_scores = self.output(hidden2) # [k, max_ant, 1]

        antecedent_scores = torch.squeeze(antecedent_scores, 2)  # [k, max_ant]
        # [k, max_ant]
        # Add to score the score of the corresponding proform
        antecedent_scores += top_span_mention_scores.unsqueeze(1) + top_span_mention_scores[antecedents]
        return antecedent_scores, antecedent_emb, pair_emb  # [k, max_ant]








