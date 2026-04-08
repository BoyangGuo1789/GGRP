import torch
from torch import nn
from typing import Dict

import ot


class PromptOTModule(nn.Module):
    def __init__(self, eps: float = 0.05, max_iter: int = 100):
        super().__init__()
        self.eps = eps
        self.max_iter = max_iter

    @torch.no_grad()
    def compute_transport(
        self,
        img_feats: torch.Tensor,
        text_feats: torch.Tensor,
    ) -> torch.Tensor:
        sim = img_feats @ text_feats.t()
        sim = sim.double()

        batch_size, num_classes = sim.shape
        a = torch.ones((batch_size,), dtype=torch.float64, device=sim.device) / batch_size
        b = torch.ones((num_classes,), dtype=torch.float64, device=sim.device) / num_classes

        coupling = ot.sinkhorn(
            a,
            b,
            M=-sim,
            reg=self.eps,
            numItermax=self.max_iter,
            stopThr=1e-6,
        )

        row_sum = coupling.sum(dim=1, keepdim=True)
        transport = coupling / (row_sum + 1e-12)
        return transport.to(img_feats.dtype)

    @torch.no_grad()
    def get_pseudo_labels(self, Q: torch.Tensor) -> torch.LongTensor:
        return torch.argmax(Q, dim=1).long()

    @torch.no_grad()
    def get_confidence(self, Q: torch.Tensor) -> Dict[str, torch.Tensor]:
        Q = Q.float()
        q_max = Q.max(dim=1).values
        top2 = torch.topk(Q, k=2, dim=1)
        margin = top2.values[:, 0] - top2.values[:, 1]
        entropy = -(Q * (Q + 1e-12).log()).sum(dim=1)
        return {"q_max": q_max, "margin": margin, "entropy": entropy}
