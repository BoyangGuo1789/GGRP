import os.path as osp

import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, Tuple

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from utils import *
from dassl.utils import (
    MetricMeter, AverageMeter
)
import datetime
import time
import copy
from .prompt_ot_module import PromptOTModule
from .teacher_branch import OTTeacher

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'GGRP',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.GGRP.N_CTX
        ctx_init = cfg.TRAINER.GGRP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            if cfg.TRAINER.GGRP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.GGRP.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,
                    ctx,
                    suffix,
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,
                        ctx_i_half1,
                        class_i,
                        ctx_i_half2,
                        suffix_i,
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,
                        class_i,
                        ctx_i,
                        suffix_i,
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts

class GeneralizedCrossEntropy(nn.Module):
    def __init__(self, q: float = 0.7) -> None:
        super().__init__()
        self.q = q
        self.epsilon = 1e-6
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = self.softmax(input)
        p = p[torch.arange(p.shape[0]), target]
        p += self.epsilon
        loss = (1 - p ** self.q) / self.q
        return torch.mean(loss)

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits

    @torch.no_grad()
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    @torch.no_grad()
    def encode_text(self) -> torch.Tensor:
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features


@TRAINER_REGISTRY.register()
class GGRP(TrainerX):
    def __init__(self, cfg):
        self.use_prompt_ot = cfg.TRAINER.GGRP.ENABLE_PROMPT_OT
        self.audit_enabled = cfg.TRAINER.GGRP.AUDIT
        self.audit_only_first = cfg.TRAINER.GGRP.AUDIT_ONLY_FIRST
        self.audit_done = False
        self.audit_grad_done = False
        self.audit_tol = cfg.TRAINER.GGRP.AUDIT_TOL
        self.smoke_test = cfg.TRAINER.GGRP.SMOKE_TEST
        self.smoke_max_epoch = cfg.TRAINER.GGRP.SMOKE_MAX_EPOCH
        self.smoke_steps = cfg.TRAINER.GGRP.SMOKE_STEPS
        self.teacher_momentum = cfg.TRAINER.GGRP.TEACHER_EMA
        self.conf_high = cfg.TRAINER.GGRP.TEACHER_CONF_HIGH
        self.conf_low = cfg.TRAINER.GGRP.TEACHER_CONF_LOW
        self.ce_target = cfg.TRAINER.GGRP.CE_TARGET

        super().__init__(cfg)
        self.GCE_loss = GeneralizedCrossEntropy(q=1.0)
        self.num_equal = []
        self.confident_rate = []
        self.clean_rate  = []

        self.best_acc = -1
        self.best_epoch = -1
        self.test_acc = []

    def check_cfg(self, cfg):
        assert cfg.TRAINER.GGRP.PREC in ["fp16", "fp32", "amp"]
        assert cfg.TRAINER.GGRP.TEACHER_CONF_HIGH > cfg.TRAINER.GGRP.TEACHER_CONF_LOW
        assert 0 <= cfg.TRAINER.GGRP.TEACHER_EMA < 1
        assert cfg.TRAINER.GGRP.SMOKE_STEPS > 0
        assert cfg.TRAINER.GGRP.AUDIT_TOL > 0

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.GGRP.PREC == "fp32" or cfg.TRAINER.GGRP.PREC == "amp":
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        if self.use_prompt_ot:
            self.ot_module = PromptOTModule()
            self.teacher_model = copy.deepcopy(self.model)
            for p in self.teacher_model.parameters():
                p.requires_grad_(False)
            self.teacher_model.to(self.device)
            self.teacher = OTTeacher(self.teacher_model, self.ot_module)
            self.teacher.to(self.device)

        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.GGRP.PREC == "amp" else None


    def _sync_teacher_with_student(self):
        if not self.use_prompt_ot:
            return
        self.teacher_model.load_state_dict(self.model.state_dict(), strict=False)

    def _update_teacher(self):
        if (not self.use_prompt_ot) or self.teacher_momentum <= 0:
            return
        with torch.no_grad():
            for t_param, s_param in zip(self.teacher_model.prompt_learner.parameters(),
                                        self.model.prompt_learner.parameters()):
                t_param.data.mul_(self.teacher_momentum).add_(
                    s_param.data * (1 - self.teacher_momentum)
                )
            student_buffers = dict(self.model.prompt_learner.named_buffers())
            for name, t_buf in self.teacher_model.prompt_learner.named_buffers():
                if name in student_buffers:
                    t_buf.copy_(student_buffers[name])

    def before_train(self):
        super().before_train()
        self.audit_done = False
        self.audit_grad_done = False
        if self.use_prompt_ot:
            self._sync_teacher_with_student()
        if self.smoke_test:
            self.max_epoch = min(self.max_epoch, self.smoke_max_epoch)
            print(f"启用 smoke test：每个 epoch 最多 {self.smoke_steps} iter，总 epoch 上限 {self.max_epoch}")

    def after_train(self):
        if self.smoke_test:
            print("Finish training (smoke test)")
            elapsed = round(time.time() - self.time_start)
            elapsed = str(datetime.timedelta(seconds=elapsed))
            print(f"Elapsed: {elapsed}")
            self.close_writer()
            print("SMOKE TEST PASSED")
            return
        super().after_train()

    def forward_backward_ce(self, batch):
        image, label, gt_label = self.parse_batch_train(batch)
        
        prec = self.cfg.TRAINER.GGRP.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image)
            loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss_x": loss.item(),
            "acc_x": compute_accuracy(output, label)[0].item(),
        }

        return loss_summary
    
    def forward_backward_mae(self, batch):
        image, label, gt_label = self.parse_batch_train(batch)
        
        prec = self.cfg.TRAINER.GGRP.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = self.GCE_loss(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image)
            loss = self.GCE_loss(output, label)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss_u": loss.item(),
            "acc_u": compute_accuracy(output, label)[0].item(),
        }

        return loss_summary

    def compute_ts_loss(
        self,
        logits: torch.Tensor,
        noisy_labels: torch.Tensor,
        teacher_labels: torch.Tensor,
        q_max: torch.Tensor,
        margin: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            margin = margin.to(logits.dtype)
            w = torch.zeros_like(margin)
            teacher_mask = margin >= self.conf_low
            high_minus_low = max(self.conf_high - self.conf_low, 1e-6)
            w[teacher_mask] = (margin[teacher_mask] - self.conf_low) / high_minus_low
            w = w.clamp(0.0, 1.0)

        ce_target = torch.where(teacher_mask, teacher_labels, noisy_labels)
        ce_teacher = F.cross_entropy(
            logits,
            ce_target,
            reduction="none"
        )

        probs = F.softmax(logits, dim=1)
        batch_indices = torch.arange(logits.size(0), device=logits.device)
        noisy_probs = probs[batch_indices, noisy_labels]
        mae_noisy = 2.0 * (1.0 - noisy_probs)

        per_sample_loss = w * ce_teacher + (1.0 - w) * mae_noisy
        loss = per_sample_loss.mean()

        stats: Dict[str, torch.Tensor] = {
            "w_mean": w.mean().detach(),
            "ce_mean": ce_teacher.mean().detach(),
            "mae_mean": mae_noisy.mean().detach(),
            "teacher_mask_ratio": teacher_mask.float().mean().detach(),
            "q_max_mean": q_max.mean().detach(),
            "margin_mean": margin.mean().detach(),
        }
        return loss, stats, w, teacher_mask

    def audit_self_check(
        self,
        images: torch.Tensor,
        logits: torch.Tensor,
        teacher_out: Dict[str, torch.Tensor],
        noisy_labels: torch.Tensor,
        teacher_labels: torch.Tensor,
        weights: torch.Tensor,
        loss: torch.Tensor,
    ):
        if not self.audit_enabled:
            return
        if self.audit_only_first and self.audit_done:
            return

        B, C = logits.shape
        device = self.device
        q = teacher_out["Q"]
        q_max = teacher_out["q_max"]

        def _check_device(name, tensor):
            if tensor.device.type != device.type:
                raise RuntimeError(f"{name} 的 device 不一致: 当前 {tensor.device}, 期望 {device}")

        def _check_shape(name, tensor, expected):
            if list(tensor.shape) != list(expected):
                raise RuntimeError(f"{name} 的 shape 不匹配: 实际 {list(tensor.shape)}, 期望 {list(expected)}")

        def _check_dtype(name, tensor, allowed):
            if tensor.dtype not in allowed:
                raise RuntimeError(f"{name} 的 dtype 不符合预期: 实际 {tensor.dtype}, 允许 {allowed}")

        def _check_finite(name, tensor):
            if not torch.isfinite(tensor).all():
                raise RuntimeError(f"{name} 中含有 NaN/Inf")

        tensors_to_check = {
            "images": images,
            "logits": logits,
            "teacher_Q": q,
            "weights": weights,
            "loss": loss.detach(),
        }
        for name, tensor in tensors_to_check.items():
            _check_device(name, tensor)
            _check_finite(name, tensor)

        _check_shape("teacher_Q", q, (B, C))
        _check_shape("teacher_labels", teacher_labels, (B,))
        _check_shape("noisy_labels", noisy_labels, (B,))
        _check_shape("weights", weights, (B,))
        _check_shape("q_max", q_max, (B,))

        _check_dtype("logits", logits, [torch.float16, torch.float32, torch.bfloat16])
        _check_dtype("teacher_Q", q, [torch.float16, torch.float32, torch.bfloat16])
        _check_dtype("teacher_labels", teacher_labels, [torch.int64])
        _check_dtype("noisy_labels", noisy_labels, [torch.int64])

        if torch.any(q < -1e-6):
            raise RuntimeError("OT 矩阵包含负值，需检查 OT 模块输出。")
        row_sum = q.sum(dim=1)
        if not torch.allclose(row_sum, torch.ones_like(row_sum), atol=self.audit_tol, rtol=0):
            raise RuntimeError(f"OT 每行和偏离 1，最大偏差 {torch.max(torch.abs(row_sum - 1)).item():.4f}")
        col_sum = q.sum(dim=0)
        if torch.any(col_sum <= 0):
            raise RuntimeError("OT 存在列和为 0，说明某些 prompt 无分配。")

        if torch.any(weights < -1e-3) or torch.any(weights > 1 + 1e-3):
            raise RuntimeError("样本权重超出 [0,1] 范围，请检查置信度映射。")
        if torch.any(q_max < -1e-3) or torch.any(q_max > 1 + 1e-3):
            raise RuntimeError("q_max 超出 [0,1] 范围，请检查 OT 归一化。")

        if "img_feats" in teacher_out:
            _check_shape("img_feats", teacher_out["img_feats"], (B, teacher_out["img_feats"].shape[1]))
            _check_device("img_feats", teacher_out["img_feats"])
        if "text_feats" in teacher_out:
            _check_shape("text_feats", teacher_out["text_feats"], (teacher_out["text_feats"].shape[0], teacher_out["text_feats"].shape[1]))
            _check_device("text_feats", teacher_out["text_feats"])

    def _audit_gradients(self):
        if not self.audit_enabled:
            return
        if self.audit_only_first and self.audit_grad_done:
            return

        student_params = [p for p in self.model.prompt_learner.parameters() if p.requires_grad]
        student_has_grad = any(
            (p.grad is not None) and torch.isfinite(p.grad).all() and (p.grad.abs().sum() > 0)
            for p in student_params
        )
        if not student_has_grad:
            raise RuntimeError("学生分支梯度为空或全部为 0，检查损失和优化器。")

        if self.use_prompt_ot:
            teacher_grads = [
                p.grad for p in self.teacher_model.parameters()
                if p.requires_grad
            ]
            teacher_has_grad = any(
                (g is not None) and torch.isfinite(g).all() and (g.abs().sum() > 0)
                for g in teacher_grads
            )
            if teacher_has_grad:
                raise RuntimeError("教师分支存在梯度，应确保 forward 使用 no_grad/EMA。")

        self.audit_grad_done = True
        self.audit_done = True

    def forward_backward_ts(self, batch):
        image, noisy_labels, gt_labels = self.parse_batch_train(batch)

        with torch.no_grad():
            teacher_out = self.teacher(image, return_features=self.audit_enabled)
            teacher_labels = teacher_out["pseudo_labels"]
            q_max = teacher_out["q_max"]
            margin = teacher_out["margin"]

        prec = self.cfg.TRAINER.GGRP.PREC

        if prec == "amp":
            with autocast():
                logits = self.model(image)
                loss, stats, weights, teacher_mask = self.compute_ts_loss(
                    logits=logits,
                    noisy_labels=noisy_labels,
                    teacher_labels=teacher_labels,
                    q_max=q_max,
                    margin=margin,
                )
            if self.audit_enabled and (not self.audit_done):
                self.audit_self_check(image, logits, teacher_out, noisy_labels, teacher_labels, weights, loss)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optim)
            if self.audit_enabled and (not self.audit_grad_done):
                self._audit_gradients()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            logits = self.model(image)
            loss, stats, weights, teacher_mask = self.compute_ts_loss(
                logits=logits,
                noisy_labels=noisy_labels,
                teacher_labels=teacher_labels,
                q_max=q_max,
                margin=margin,
            )
            if self.audit_enabled and (not self.audit_done):
                self.audit_self_check(image, logits, teacher_out, noisy_labels, teacher_labels, weights, loss)
            self.model_backward_and_update(loss)
            if self.audit_enabled and (not self.audit_grad_done):
                self._audit_gradients()

        self._update_teacher()

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            teacher_noisy_agree = (teacher_labels == noisy_labels).float().mean()
            student_teacher_agree = (preds == teacher_labels).float().mean()
            student_noisy_agree = (preds == noisy_labels).float().mean()
            w_mean, w_min, w_max = weights.mean(), weights.min(), weights.max()
            q_mean, q_min, q_max_val = q_max.mean(), q_max.min(), q_max.max()
            entropy_mean = teacher_out["entropy"].mean()
            margin_mean = margin.mean()
            teacher_mask_ratio = stats.get("teacher_mask_ratio", torch.tensor(0.0))

        loss_summary = {
            "loss_ts": loss.item(),
            "acc_noisy": compute_accuracy(logits, noisy_labels)[0].item(),
            "acc_teacher": compute_accuracy(logits, teacher_labels)[0].item(),
            "w_mean": w_mean.item(),
            "w_min": w_min.item(),
            "w_max": w_max.item(),
            "q_max_mean": q_mean.item(),
            "q_max_min": q_min.item(),
            "q_max_max": q_max_val.item(),
            "teacher_noisy_agree": teacher_noisy_agree.item(),
            "student_teacher_agree": student_teacher_agree.item(),
            "student_noisy_agree": student_noisy_agree.item(),
            "entropy_mean": entropy_mean.item(),
            "margin_mean": margin_mean.item(),
            "ce_mean": stats["ce_mean"].item(),
            "mae_mean": stats["mae_mean"].item(),
            "teacher_mask_ratio": teacher_mask_ratio.item(),
        }
        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        gt_label = batch["gttarget"]
        input = input.to(self.device)
        label = label.to(self.device)
        gt_label = gt_label.to(self.device)
        return input, label, gt_label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            self._models[name].load_state_dict(state_dict, strict=False)
        if self.use_prompt_ot:
            self._sync_teacher_with_student()

    def before_epoch(self):
        self._epoch_wall_start = time.time()
        if self.use_prompt_ot:
            if not self.audit_only_first:
                self.audit_done = False
                self.audit_grad_done = False
            return
        cfg = self.cfg
        if cfg.DATASET.USE_OT == True:
            reg_feat = cfg.DATASET.REG_FEAT
            reg_lab = cfg.DATASET.REG_LAB
            curriclum_epoch = cfg.DATASET.CURRICLUM_EPOCH
            begin_rate = cfg.DATASET.BEGIN_RATE
            curriclum_mode = cfg.DATASET.CURRICLUM_MODE
            Pmode = cfg.DATASET.PMODE
            reg_e = cfg.DATASET.REG_E

            if self.epoch < curriclum_epoch:
                budget, pho = curriculum_scheduler(self.epoch, curriclum_epoch, begin=begin_rate,end=1,mode=curriclum_mode)
            else:
                budget, pho = 1., 1.

            with torch.no_grad():
                pseudo_labels1, noisy_labels, gt_labels, selected_mask, conf1, argmax_plabels = OT_PL(self.model, 
                        self.train_loader_x, num_class=cfg.DATASET.num_class, batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE, budget=budget, reg_feat=reg_feat, 
                        reg_lab=reg_lab,Pmode=Pmode, reg_e=reg_e, load_all=True)

                print("before epoch:data num:", len(gt_labels))
                print("before epoch:different number:", np.sum(gt_labels.cpu().numpy() != argmax_plabels.cpu().numpy()))

                conf_l_mask, conf_u_mask, lowconf_u_mask = get_masks(argmax_plabels, noisy_labels, None, selected_mask)
                selected_rate_conf_l, selected_rate_conf_u, selected_rate_lowconf_u = output_selected_rate(conf_l_mask, conf_u_mask, lowconf_u_mask)
                print("confident_label rate",selected_rate_conf_l)
                unlabeled_mask1 = torch.logical_or(conf_u_mask, lowconf_u_mask)

            if np.sum(conf_l_mask.cpu().numpy()) > 0:
                mask = conf_l_mask.cpu().numpy() 
                self.mask2 = unlabeled_mask1.cpu().numpy()
                pred_idx = mask.nonzero()[0]
                pred_idx2 = self.mask2.nonzero()[0]
                conf = conf1.cpu().numpy()
                plabel = argmax_plabels.cpu().numpy()

                self.tmp_train_loader_x = copy.deepcopy(self.train_loader_x)
                self.train_loader_u = copy.deepcopy(self.train_loader_x)
                

                print("before: len(self.train)",len(self.train_loader_x.dataset.data_source))
                print("before: len of confident samples",len(pred_idx))


                count11=0
                count12=0
                count21=0
                count22=0
                for i in range(len(self.train_loader_x.dataset.data_source)):
                    if mask[i]== True:
                        if plabel[i] == gt_labels[i]:
                            count11 += 1
                        else:
                            count12 += 1
                    elif self.mask2[i]== True:
                        if plabel[i] == gt_labels[i]:
                            count21 += 1
                        else:
                            count22 += 1
                print(f"clean true:{count11}")
                print(f"clean false:{count12}")
                clean_rate=count11/(count11+count12)
                print(f"clean_rate:{clean_rate}")
                self.clean_rate.append(clean_rate)
                print(f"noisy true:{count21}")
                print(f"noisy false:{count22}")

                if self.epoch == 99:
                    print("all clean rate: ", self.clean_rate)

                for index in sorted(pred_idx2,reverse = True):
                    del self.train_loader_x.dataset.data_source[index]
                print("after delete: len(clean_dataset)",len(self.train_loader_x.dataset.data_source))

                for index in sorted(pred_idx,reverse = True):
                    del self.train_loader_u.dataset.data_source[index]
                print("after delete: len(noisy_dataset)",len(self.train_loader_u.dataset.data_source))

    def _run_epoch_promptot(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        total_batches = len(self.train_loader_x)
        iter_limit = min(total_batches, self.smoke_steps) if self.smoke_test else total_batches
        self.num_batches = iter_limit

        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_x):
            if self.smoke_test and self.batch_idx >= iter_limit:
                break

            data_time.update(time.time() - end)
            loss_summary = self.forward_backward_ts(batch)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = iter_limit < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = iter_limit - self.batch_idx - 1
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                info = [
                    f"epoch [{self.epoch + 1}/{self.max_epoch}]",
                    f"batch [{self.batch_idx + 1}/{iter_limit}]",
                    f"time {batch_time.val:.3f} ({batch_time.avg:.3f})",
                    f"data {data_time.val:.3f} ({data_time.avg:.3f})",
                    f"{losses}",
                    f"lr {self.get_current_lr():.4e}",
                    f"eta {eta}",
                ]
                print(" ".join(info))

            n_iter = self.epoch * max(iter_limit, 1) + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train_ts/" + name, meter.avg, n_iter)
            self.write_scalar("train_ts/lr", self.get_current_lr(), n_iter)

            end = time.time()

        self.update_lr()

    def run_epoch(self):
        if self.use_prompt_ot:
            return self._run_epoch_promptot()
        self.set_model_mode("train")
        losses_x = MetricMeter()  
        losses_u = MetricMeter()  
        batch_time = AverageMeter()
        data_time = AverageMeter()

        if self.train_loader_x is not None:
            train_loader_x_iter = iter(self.train_loader_x)
            len_train_loader_x = len(self.train_loader_x)
        else:
            len_train_loader_x = 0

        if self.train_loader_u is not None:
            train_loader_u_iter = iter(self.train_loader_u)
            len_train_loader_u = len(self.train_loader_u)
        else:
            len_train_loader_u = 0

        self.num_batches_x = len_train_loader_x
        self.num_batches_u = len_train_loader_u

        end = time.time()
        
        for self.batch_idx in range(self.num_batches_x):
            try:
                batch_x = next(train_loader_x_iter)
                data_time.update(time.time() - end)
                loss_summary_x = self.forward_backward_ce(batch_x)
                losses_x.update(loss_summary_x)
            except StopIteration:
                break  

            batch_time.update(time.time() - end)

            if (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0 or self.num_batches_x < self.cfg.TRAIN.PRINT_FREQ:
                eta_seconds = batch_time.avg * (self.num_batches_x - self.batch_idx - 1)
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = [
                    f"epoch [{self.epoch + 1}/{self.max_epoch}]",
                    f"batch [{self.batch_idx + 1}/{self.num_batches_x}]",
                    f"time {batch_time.val:.3f} ({batch_time.avg:.3f})",
                    f"data {data_time.val:.3f} ({data_time.avg:.3f})",
                    f"loss_x {losses_x}",
                    f"lr {self.get_current_lr():.4e}",
                    f"eta {eta}"
                ]
                print(" ".join(info))

            n_iter = self.epoch * (self.num_batches_x + self.num_batches_u) + self.batch_idx
            for name, meter in losses_x.meters.items():
                self.write_scalar("train_x/" + name, meter.avg, n_iter)

            end = time.time()

        for self.batch_idx in range(self.num_batches_u):
            try:
                batch_u = next(train_loader_u_iter)
                data_time.update(time.time() - end)
                loss_summary_u = self.forward_backward_mae(batch_u)
                losses_u.update(loss_summary_u)
            except StopIteration:
                break

            batch_time.update(time.time() - end)

            if (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0 or self.num_batches_u < self.cfg.TRAIN.PRINT_FREQ:
                eta_seconds = batch_time.avg * (self.num_batches_u - self.batch_idx - 1)
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = [
                    f"epoch [{self.epoch + 1}/{self.max_epoch}]",
                    f"batch [{self.batch_idx + 1}/{self.num_batches_u}]",
                    f"time {batch_time.val:.3f} ({batch_time.avg:.3f})",
                    f"data {data_time.val:.3f} ({data_time.avg:.3f})",
                    f"loss_u {losses_u}",
                    f"lr {self.get_current_lr():.4e}",
                    f"eta {eta}"
                ]
                print(" ".join(info))

            n_iter = self.epoch * (self.num_batches_x + self.num_batches_u) + self.batch_idx
            for name, meter in losses_u.meters.items():
                self.write_scalar("train_u/" + name, meter.avg, n_iter)

            end = time.time()

        self.update_lr()

    def after_epoch(self):
        if hasattr(self, "_epoch_wall_start"):
            epoch_wall = time.time() - self._epoch_wall_start
            readable = str(datetime.timedelta(seconds=int(epoch_wall)))
            print(f"epoch [{self.epoch + 1}/{self.max_epoch}] wall time {readable} ({epoch_wall:.3f}s)")

        if self.use_prompt_ot:
            return super().after_epoch()
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = (
            (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
            if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False
        )

        if do_test and self.cfg.TEST.FINAL_MODEL == "best_val":
            curr_result = self.test(split="val")
            is_best = curr_result > self.best_result
            if is_best:
                self.best_result = curr_result
                self.save_model(
                    self.epoch,
                    self.output_dir,
                    val_result=curr_result,
                    model_name="model-best.pth.tar"
                )
        
        if meet_checkpoint_freq or last_epoch:
            self.save_model(self.epoch, self.output_dir)
        
        if self.cfg.DATASET.USE_OT == True:
            self.train_loader_x = copy.deepcopy(self.tmp_train_loader_x)
            self.train_loader_u = copy.deepcopy(self.tmp_train_loader_x)
            print("after epoch: len(clean dataset)", len(self.train_loader_x.dataset.data_source))
            print("after epoch: len(noisy dataset)", len(self.train_loader_u.dataset.data_source))
