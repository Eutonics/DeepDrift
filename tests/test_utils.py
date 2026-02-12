import torch
import numpy as np
from deepdrift.utils.pooling import cls_pool, sparse_sample
from deepdrift.utils.stats import compute_iqr_threshold

def test_cls_pool():
    x = torch.randn(2, 5, 10) # [B, Seq, Dim]
    pooled = cls_pool(x)
    assert pooled.shape == (2, 10)
    assert torch.allclose(pooled, x[:, 0, :])

def test_sparse_sample():
    x = torch.randn(2, 100)
    sampled = sparse_sample(x, n_channels=10)
    assert sampled.shape == (2, 10)

def test_iqr_threshold():
    vels = np.array([1, 1.1, 1.2, 0.9, 1.0, 5.0]) # 5.0 is outlier
    # Q75 of [0.9, 1, 1, 1.1, 1.2, 5]
    # Q25=1, Q75=1.15 approx
    threshold = compute_iqr_threshold(vels)
    assert threshold > 1.2
    assert threshold < 5.0
