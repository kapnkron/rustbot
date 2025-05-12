import torch

def cuda_add(x, y):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a = torch.tensor([x], device=device)
    b = torch.tensor([y], device=device)
    result = a + b
    return result.cpu().numpy().tolist() 