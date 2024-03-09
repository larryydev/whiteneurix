import torch.nn as nn

class WhiteBalanceModel(nn.Module):
    def __init__(self, in_, out_):
        super(WhiteBalanceModel, self).__init__()

        