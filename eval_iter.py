from torchtext import data

class EvalIterator(data.Iterator):
    def create_batches(self):
        self.batches = custom_batch(self.data())


def custom_batch(data):
    """Yield elements from data where each batch represents a sentence."""
    minibatch = []
    old_doc_id = None

    for ex in data:
        minibatch.append(ex)

        if not old_doc_id:
            old_doc_id = ex.doc_id

        if old_doc_id != ex.doc_id:
            old_doc_id = ex.doc_id
            yield minibatch[:-1]
            minibatch = minibatch[-1:]

    if minibatch:
        yield minibatch