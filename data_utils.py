def load_lm_embeddings_for_sentence(
        lm_file,
        doc_key,
        sent_key,
        transpose):
    """
    Load LM embeddings for a given sentence.
    """
    # TODO preprocessed embeddings that needs to be adjusted with different datasets
    print(sent_key)
    file_key = doc_key.replace("/", ":")
    print(file_key)

    group = lm_file[file_key]
    print(group)
    if sent_key is not None:
        sentence = group[sent_key][...]
    else:
        sentence = group[...]

    print(sentence)
    exit()
    if transpose:
        return sentence.transpose(1, 2, 0)
    else:
        return sentence
