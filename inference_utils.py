import torch

_CORE_ARGS = {"ARG0": 1, "ARG1": 2, "ARG2": 4, "ARG3": 8, "ARG4": 16, "ARG5": 32, "ARGA": 64,
              "A0": 1, "A1": 2, "A2": 4, "A3": 8, "A4": 16, "A5": 32, "AA": 64}


# One-stop decoder for all the tasks.
def mtl_decode(sentences, predict_dict, ner_labels_inv, rel_labels_inv, config):
    predictions = {}

    # Decode sentence-level tasks.
    num_sentences = len(sentences)
    if "srl_scores" in predict_dict:
        predictions["srl"] = [{} for _ in range(num_sentences)]
    if "ner_scores" in predict_dict:
        predictions["ner"] = [{} for _ in range(num_sentences)]
    if "rel_scores" in predict_dict:
        predictions["rel"] = [[] for _ in range(num_sentences)]

    # Sentence-level predictions.
    for i in range(num_sentences):
        if "rel" in predictions:
            # Num entities per sentence
            num_ents = predict_dict["num_entities"][i]
            ent_starts = predict_dict["entity_starts"][i]
            ent_ends = predict_dict["entity_ends"][i]
            for j in range(num_ents):
                for k in range(num_ents):
                    pred = predict_dict["rel_labels"][i, j, k]
                    if pred > 0:
                        predictions["rel"][i].append([
                            ent_starts[j], ent_ends[j], ent_starts[k], ent_ends[k],
                            rel_labels_inv[pred]])
        if "ner" in predictions:
            ner_spans, _, _ = _dp_decode_non_overlapping_spans(
                predict_dict["candidate_starts"][i],
                predict_dict["candidate_ends"][i],
                # [num_sentences, max_num_candidates, num_labels]
                predict_dict["ner_scores"][i],
                len(sentences[i]), ner_labels_inv, None, False)
            predictions["ner"][i] = ner_spans

    # Document-level predictions. -1 means null antecedent.
    if "antecedent_scores" in predict_dict:
        mention_spans = list(zip(predict_dict["mention_starts"], predict_dict["mention_ends"]))
        mention_to_predicted = {}
        predicted_clusters = []

        def _link_mentions(curr_span, ant_span):
            if ant_span not in mention_to_predicted:
                new_cluster_id = len(predicted_clusters)
                mention_to_predicted[ant_span] = new_cluster_id
                predicted_clusters.append([ant_span, ])
            cluster_id = mention_to_predicted[ant_span]
            if not curr_span in mention_to_predicted:
                mention_to_predicted[curr_span] = cluster_id
                predicted_clusters[cluster_id].append(curr_span)
            '''else:
              cluster_id2 = mention_to_predicted[curr_span]
              # Merge clusters.
              if cluster_id != cluster_id2:
                print "Merging clusters:", predicted_clusters[cluster_id], predicted_clusters[cluster_id2]
                for span in predicted_clusters[cluster_id2]:
                  mention_to_predicted[span] = cluster_id
                  predicted_clusters[cluster_id].append(span)
                predicted_clusters[cluster_id2] = []'''

        scores = predict_dict["antecedent_scores"]
        antecedents = predict_dict["antecedents"]
        # if config["coref_loss"] == "mention_rank":
        for i, ant_label in enumerate(torch.argmax(scores, dim=1)):
            if ant_label <= 0:
                continue
            ant_id = antecedents[i][ant_label - 1]
            assert i > ant_id
            _link_mentions(mention_spans[i], mention_spans[ant_id])
        '''else:
          for i, curr_span in enumerate(mention_spans):
            for j in range(1, scores.shape[1]):
              if scores[i][j] > 0:
                _link_mentions(curr_span, mention_spans[antecedents[i][j-1]])'''

        predicted_clusters = [tuple(sorted(pc)) for pc in predicted_clusters]
        predictions["predicted_clusters"] = predicted_clusters
        predictions["mention_to_predicted"] = {m: predicted_clusters[i] for m, i in mention_to_predicted.items()}

    # print predictions["srl"]
    return predictions


def _dp_decode_non_overlapping_spans(starts, ends, scores, max_len, labels_inv, pred_id,
                                     u_constraint=False):
    # Num labels
    num_roles = scores.shape[1]
    # Get the pred for each candidate
    labels = torch.argmax(scores, dim=1)
    spans = zip(starts, ends, range(len(starts)))
    spans = sorted(spans, key=lambda x: (x[0], x[1]))

    if u_constraint:
        f = torch.zeros([max_len + 1, 128], dtype=torch.float32) - 0.1
    else:
        f = torch.zeros([max_len + 1, 1], dtype=torch.float32) - 0.1
    f[0, 0] = 0
    states = {0: set([0])}  # A dictionary from id to list of binary core-arg states.
    pointers = {}  # A dictionary from states to (arg_id, role, prev_t, prev_rs)
    best_state = [(0, 0)]

    def _update_state(t0, rs0, t1, rs1, delta, arg_id, role):
        if f[t0][rs0] + delta > f[t1][rs1]:
            f[t1][rs1] = f[t0][rs0] + delta
            if t1 not in states:
                states[t1] = set()
            states[t1].update([rs1])
            pointers[(t1, rs1)] = (arg_id, role, t0, rs0)
            if f[t1][rs1] > f[best_state[0][0]][best_state[0][1]]:
                best_state[0] = (t1, rs1)

    for start, end, i in spans:
        start = start.item()
        end = end.item()
        # Dummy label
        assert scores[i][0] == 0
        # The extra dummy score should be same for all states, so we can safely skip arguments overlap
        # with the predicate.
        if pred_id is not None and start <= pred_id and pred_id <= end:
            continue
        r0 = labels[i].item()  # Locally best role assignment.
        # Strictly better to incorporate a dummy span if it has the highest local score.
        if r0 == 0:
            continue

        # Consider those with actual label
        r0_str = labels_inv[r0]
        # Enumerate explored states.
        t_states = [t for t in states.keys() if t <= start]
        for t in t_states:
            role_states = states[t]
            # Update states if best role is not a core arg.
            if not u_constraint or not r0_str in _CORE_ARGS:
                for rs in role_states:
                    _update_state(t, rs, end + 1, rs, scores[i][r0].item(), i, r0)
            else:
                for rs in role_states:
                    for r in range(1, num_roles):
                        if scores[i][r] > 0:
                            r_str = labels_inv[r]
                            core_state = _CORE_ARGS.get(r_str, 0)
                            # print start, end, i, r_str, core_state, rs
                            if core_state & rs == 0:
                                _update_state(t, rs, end + 1, rs | core_state, scores[i][r], i, r)
    # Backtrack to decode.
    new_spans = []
    t, rs = best_state[0]
    while (t, rs) in pointers:
        i, r, t0, rs0 = pointers[(t, rs)]
        new_spans.append((starts[i], ends[i], labels_inv[r]))
        t = t0
        rs = rs0
    # print spans
    # print new_spans[::-1]
    return new_spans[::-1]
