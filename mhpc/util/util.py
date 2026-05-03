import torch as _torch
import numpy as _np

def to_numpy(t: _torch.Tensor) -> _np.ndarray:
    return detach(t).cpu().numpy()

def detach(t: _torch.Tensor) -> _torch.Tensor:
    """
    Detach a tensor from the current computation graph.
    
    Args:
        t (torch.Tensor): The tensor to detach.
    
    Returns:
        torch.Tensor: The detached tensor.
    """
    return t.detach() if t.requires_grad else t

