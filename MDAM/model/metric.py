import torch



def mae(output, target):
    assert output.shape == target.shape
    with torch.no_grad():
        return torch.mean(torch.abs(output - target)).cpu().item()

