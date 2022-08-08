# Bidirectional Long Expressive Memory
This repository contains the implementation of Bidirectional Long Expressive Memory, which is extension of paper [Long Expressive Memory for Sequence Modeling](https://openreview.net/forum?id=vwj6aUeocyf).

## Usage:
```python
import torch
from lem import BidirectionalSeqLEM


input = torch.ones([2, 1, 784])    # Size : [B, dim, seq_len]

model = BidirectionalSeqLEM( ninp = 1, nhid = 128, dt = 0.3)

ys, zs = model(input.permute(2, 0, 1)) # ys shape : [seq_len, B, 2 * dim], zs shape : [seq_len, B, 2 * dim]

```



## References:
* https://github.com/tk-rusch/LEM
