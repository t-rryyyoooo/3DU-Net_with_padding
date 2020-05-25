from torch import nn
import torch
from torch.autograd import Variable

class WeightedCategoricalCrossEntropy(nn.Module):
    def __init__(self, *function, device):
        super(WeightedCategoricalCrossEntropy, self).__init__()
        self.function = function
        self.device = device

    def forward(self, pred, true):
        """ 
        onehot
        """
        
        eps = 10**(-9)
        result = torch.sum(true, dim=[0, 1, 2, 3])
        
        if len(self.function) == 1:
            f = "torch.{}({})".format(self.function[0], "result")
            
            result_f = eval(f)
        else:
            args = self.function[1:]
            
            arg = "result,"
            for l in range(len(args)):
                arg += args[l]
                if l != (len(args) - 1):
                    args += ","
            
            f = "torch.{}({})".format(self.function[0], arg)
            
            result_f = eval(f)
            
        
        weight = result_f / torch.sum(result_f)
        
        
        output = ((-1) * torch.sum(1 / (weight + eps) * true * torch.log(pred + eps), axis=-1)).to(self.device)

        output = output.mean().to(self.device)


        return output
