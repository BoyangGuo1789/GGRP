import torch
from torch import nn
from typing import Dict

from .prompt_ot_module import PromptOTModule


class OTTeacher(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        ot_module: PromptOTModule,
    ):
        super().__init__()
        self.model = model
        self.ot_module = ot_module

        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, images: torch.Tensor, return_features: bool = False) -> Dict[str, torch.Tensor]:
        img_feats = self.model.encode_image(images)
        text_feats = self.model.encode_text()

        Q = self.ot_module.compute_transport(img_feats, text_feats).float()

        pseudo_labels = self.ot_module.get_pseudo_labels(Q)
        conf = self.ot_module.get_confidence(Q)

        out: Dict[str, torch.Tensor] = {
            "Q": Q,
            "pseudo_labels": pseudo_labels,
            "q_max": conf["q_max"],
            "margin": conf["margin"],
            "entropy": conf["entropy"],
        }
        if return_features:
            out["img_feats"] = img_feats
            out["text_feats"] = text_feats
        return out
