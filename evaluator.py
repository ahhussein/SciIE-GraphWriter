class Evaluator:
    def __init__(self, config):
        self.config = config
        self.eval_data = None

    def evaluate(self, data, predictions, loss, official_stdout=False):
        if self.eval_data is None:
            self.eval_data, self.eval_tensors, self.coref_eval_data = data.load_eval_data()
