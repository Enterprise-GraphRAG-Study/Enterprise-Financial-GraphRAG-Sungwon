import torch

def get_device() -> torch.device:
    """
    Checks for available hardware acceleration and returns the best device.
    
    Returns:
        torch.device: The detected device (mps, cuda, or cpu).
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    print(f"Using device: {device}")
    return device