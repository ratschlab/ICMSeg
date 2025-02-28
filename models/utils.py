import numpy as np

def cosine_annealing(step: int, total_steps: int, lr_max: float, lr_min: float, 
                     warmup_steps: int = 0):
    
    assert warmup_steps >= 0

    if step < warmup_steps:
        lr = lr_max * step / warmup_steps
    else:
        lr = lr_min + (lr_max - lr_min) * 0.5 * \
            (1 + np.cos((step - warmup_steps) / \
                        (total_steps - warmup_steps) * np.pi))

    return lr


