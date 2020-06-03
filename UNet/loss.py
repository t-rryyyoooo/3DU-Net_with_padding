from torch import nn
import torch
from torch.autograd import Variable

class WeightedCategoricalCrossEntropy(nn.Module):
    def __init__(self, device):
        super(WeightedCategoricalCrossEntropy, self).__init__()
        self.device = device

    def forward(self, pred, true):
        """ 
        onehot
        """
        
        eps = 10**(-9)
        result = torch.sum(true, dim=[0, 1, 2, 3, 4])

        result_f = torch.log(result)
    
        weight = result_f / torch.sum(result_f)
        
        output = ((-1) * torch.sum(1 / (weight + eps) * true * torch.log(pred + eps), axis=1))

        output = output.mean()


        return output
