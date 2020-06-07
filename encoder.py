import json
import numpy
import torch
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)
