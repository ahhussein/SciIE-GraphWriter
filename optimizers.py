class MultipleOptimizer(object):
    def __init__(self, opt):
        self.optimizers = opt

    def zero_grad(self):
        for key, op in self.optimizers.items():
            op.zero_grad()

    def step(self, only_opt=None):
        if only_opt:
            for key in only_opt:
                self.optimizers[key].step()
            return

        for key, op in self.optimizers.items():
            op.step()