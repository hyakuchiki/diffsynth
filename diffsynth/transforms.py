import torch

class LogTransform(torch.nn.Module):
    """
    Perform log-amplitude transform on the data
    
    Args:
        factor (int): scale of the Gaussian noise. default: 1e-5
    """
    def __init__(self, clip=1e-3):
        super().__init__()
        self.clip = clip

    def __call__(self, data):
        if (self.clip == 0):
            data = torch.log1p(data)
        else:
            data = torch.log(data + self.clip)
        return data

class Permute(torch.nn.Module):
    """
    Because melspectrograms need to be permuted
    
    Args:
        permute_order (sequence): (0,1,2)
    """
    def __init__(self, permute_order):
        super().__init__()
        self.permute = permute_order

    def __call__(self, data):
        return data.permute(*self.permute)