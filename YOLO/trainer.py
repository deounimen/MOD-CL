#lambda version:
# File path: YOLO/trainer.py
# Please use this code to completely replace the contents of your file.

import torch
import torch.nn as nn
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.tal import dist2bbox, make_anchors
from .loss import YOLOLoss
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.torch_utils import de_parallel
from ultralytics.utils.plotting import plot_labels
from ultralytics.utils import DEFAULT_CFG, RANK, colorstr, LOGGER, TQDM
from .dataset import ROADYOLODataset
from .validator import YOLOValidator
from copy import copy
import numpy as np


# ===================================================================
# LambdaHead: predict K lambdas from the last feature map (Sigmoid in (0,1))
# ===================================================================
class LambdaHead(nn.Module):
    def __init__(self, input_channels: int, out_dim: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)  # [B,C,H,W] -> [B,C,1,1]
        self.flatten = nn.Flatten()          # [B,C,1,1] -> [B,C]
        self.mlp = nn.Sequential(
            nn.Linear(input_channels, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, out_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.pool(x)
        x = self.flatten(x)   # [B,C]
        return self.mlp(x)    # [B,out_dim]  each in (0,1)


# ===================================================================
# YOLOModel: dynamic λ rule injection (13 rules, single pass; positive then negative in the same head)
# ===================================================================
class YOLOModel(DetectionModel):

    def __init__(self, cfg='yolov8x.yaml', ch=3, nc=None, verbose=True, req_loss=False):
        super().__init__(cfg, ch, nc, verbose)
        self.nc = nc
        self.req_loss = req_loss
        self.num_rule_passes = 1  # single pass

        # Detection head channels (no = reg_max*4 + nc), used as input channels for LambdaHead
        no = getattr(self.model[-1], 'no', None)
        if no is None:
            reg_max = getattr(self.model[-1], 'reg_max', 16)
            no = reg_max * 4 + (nc if nc is not None else 0)

        # ====== Rule table (from your 3-layer definition, 13 rules; slots 0..12) ======
        # Notation: sign = +1 (use h[i]), sign = -1 (use 1 - h[i])
        # head_sign: +1 = smax (pull up), -1 = smin (push down)
        #
        # Class indices (ROAD-R):
        # 0:Ped 1:Car 2:Cyc 3:Mobike 4:MedVeh 5:LarVeh 6:Bus 7:EmVeh 8:TL 9:OthTL
        # 10:Red 11:Amber 12:Green 13:MovAway 14:MovTow 15:Mov 16:Brake 17:Stop
        # 18:IncatLft 19:IncatRht 20:HazLit 21:TurLft 22:TurRht 23:Ovtak 24:Wait2X
        # 25:XingFmLft 26:XingFmRht 27:Xing 28:PushObj 29:VehLane 30:OutgoLane
        # 31:OutgoCycLane 32:IncomLane 33:IncomCycLane 34:Pav 35:LftPav 36:RhtPav
        # 37:Jun 38:xing 39:BusStop 40:parking
        self.rules = [
            # ---------- Layer 1 (8) ----------
            {"head_idx": 23, "head_sign": -1, "body": [(17, -1)], "lambda_slot": 0},          # n23 :- n17
            {"head_idx": 18, "head_sign": -1, "body": [(0, +1)],  "lambda_slot": 1},          # n18 :- 0
            {"head_idx": 12, "head_sign": -1, "body": [(10, +1)], "lambda_slot": 2},          # n12 :- 10
            {"head_idx": 28, "head_sign": -1, "body": [(0, -1)],  "lambda_slot": 3},          # n28 :- n0
            {"head_idx": 36, "head_sign": +1, "body": [(0, +1), (35, -1)], "lambda_slot": 4}, # 36 :- 0 n35
            {"head_idx": 20, "head_sign": -1, "body": [(32, -1)], "lambda_slot": 5},          # n20 :- n32
            {"head_idx": 28, "head_sign": -1, "body": [(1, +1)],  "lambda_slot": 6},          # n28 :- 1
            {"head_idx": 38, "head_sign": -1, "body": [(30, +1)], "lambda_slot": 7},          # n38 :- 30

            # ---------- Layer 2 (4) ----------
            {"head_idx": 24, "head_sign": +1,
             "body": [(0,+1),(13,-1),(14,-1),(15,-1),(17,-1),(21,-1),(22,-1),(25,-1),(26,-1),(27,-1),(28,-1)],
             "lambda_slot": 8},                                                                # 24 :- 0 n13 ... n28
            {"head_idx": 8,  "head_sign": +1, "body": [(9, -1), (12, +1)], "lambda_slot": 9},  # 8 :- n9 12
            {"head_idx": 7,  "head_sign": -1, "body": [(28, +1)],         "lambda_slot": 10}, # n7 :- 28
            {"head_idx": 8,  "head_sign": +1, "body": [(9, -1), (10, +1)], "lambda_slot": 11}, # 8 :- n9 10

            # ---------- Layer 3 (1) ----------
            {"head_idx": 4,  "head_sign": +1,
             "body": [(1,-1),(2,-1),(3,-1),(5,-1),(6,-1),(7,-1),(18,+1)],
             "lambda_slot": 12},                                                               # 4 :- n1 n2 n3 n5 n6 n7 18
        ]

        # Number of dynamic λ
        self.num_rules = len(self.rules)  # 13
        # Build output for LambdaHead
        self.lambda_head = LambdaHead(input_channels=no, out_dim=self.num_rules)

        # ---------- Training stabilization & diagnostics ----------
        # Global scale and linear warm-up (avoid overly strong rules at the beginning)
        self.lambda_global_scale = 0.30
        self.lambda_warmup_batches = 2000
        # Per-rule additional scale (downscale “problematic” rules if needed), default all 1
        self.register_buffer("slot_scale", torch.ones(self.num_rules))
        # Training step counter & sparse diagnostic print
        self._global_step = 0
        self._dbg_every = 200

    # ------------------------- Utility: split/merge prediction tensors -------------------------
    def _split_pred_tensor(self, pred_tensor, nc):
        info = {'type': None, 'layout': None, 'shape': tuple(pred_tensor.shape)}
        if pred_tensor.ndim == 4:
            # Training mode [B,C,H,W]
            b, c, h, w = pred_tensor.shape
            pt = pred_tensor.permute(0, 2, 3, 1)  # [B,H,W,C]
            reg_max = getattr(self.model[-1], 'reg_max', 16)
            box_ch = reg_max * 4
            box_part = pt[..., :box_ch]          # [B,H,W,box_ch]
            cls_part = pt[..., box_ch:]          # [B,H,W,nc] logits
            info.update(type='4d', layout='bhwc', reg_max=reg_max)
            return box_part, cls_part, info

        elif pred_tensor.ndim == 3:
            B, A, B_or_N = pred_tensor.shape
            if A == 4 + nc:
                pt = pred_tensor.transpose(1, 2).contiguous()  # [B,N,4+C]
                box_part = pt[..., :4]
                cls_part = pt[..., 4:]                          # [B,N,C] probs (inference/validation)
                info.update(type='3d', layout='b_n_4c_channels_first')
                return box_part, cls_part, info
            elif B_or_N == 4 + nc:
                box_part = pred_tensor[..., :4]
                cls_part = pred_tensor[..., 4:]                  # [B,N,C] probs
                info.update(type='3d', layout='b_n_4c_channels_last')
                return box_part, cls_part, info
            else:
                info.update(type='unknown')
                return None, None, info
        else:
            info.update(type='unknown')
            return None, None, info

    @staticmethod
    def _merge_pred_tensor(box_part, cls_part, info):
        if info['type'] == '4d':
            recomb = torch.cat([box_part, cls_part], dim=-1)
            return recomb.permute(0, 3, 1, 2).contiguous()
        elif info['type'] == '3d':
            recomb = torch.cat([box_part, cls_part], dim=-1]  # [B,N,4+C]
            if info['layout'] == 'b_n_4c_channels_first':
                return recomb.transpose(1, 2).contiguous()     # [B,4+C,N]
            else:
                return recomb                                   # [B,N,4+C]
        else:
            return torch.cat([box_part, cls_part], dim=-1)

    # ------------------------- Rule logic (same head: positive first, then negative; single pass) -------------------------
    def _apply_rules_prob(self, cls_prob, lambdas, rules):
        """
        cls_prob: [B,H,W,C] or [B,N,C] — probability domain
        lambdas:  [B,1,1,K] or [B,1,K] — aligned with rules order
        Phase-1: aggregate all positive heads (smax) -> mid[head]
        Phase-2: on mid[head], aggregate all negative heads (smin) -> out[..., head]
        All body literals are read from the same src to avoid intra-layer chaining.
        """
        # Numerical safeguard: replace any nan/inf with boundary values to avoid exploding logits downstream
        src = torch.nan_to_num(cls_prob, nan=0.0, posinf=1.0, neginf=0.0).clamp(1e-6, 1.0 - 1e-6)
        out = src.clone()
        one = torch.ones(1, dtype=src.dtype, device=src.device)

        # Pre-compute body scores for each rule (from src)
        pos_rules, neg_rules = {}, {}
        for r in rules:
            head = r["head_idx"]; head_sign = r["head_sign"]; body = r["body"]; ls = r["lambda_slot"]
            a = None
            for (idx, sign) in body:
                lit = src[..., idx] if sign == +1 else (one - src[..., idx])
                a = lit if a is None else torch.minimum(a, lit)
            (pos_rules if head_sign == +1 else neg_rules).setdefault(head, []).append((ls, a))

        heads = set(list(pos_rules.keys()) + list(neg_rules.keys()))
        mid = {}

        # Phase-1: positive heads (smax)
        for h in heads:
            old = src[..., h]
            if h in pos_rules:
                preds = []
                for (ls, a) in pos_rules[h]:
                    lam = lambdas[..., ls]
                    cand = torch.maximum(old, a)
                    preds.append((one - lam) * old + lam * cand)
                agg_pos = preds[0]
                for t in preds[1:]:
                    agg_pos = torch.maximum(agg_pos, t)
                mid[h] = agg_pos
            else:
                mid[h] = old

        # Phase-2: negative heads (smin)
        for h in heads:
            base = mid[h]
            if h in neg_rules:
                preds = []
                for (ls, a) in neg_rules[h]:
                    lam = lambdas[..., ls]
                    cand = torch.minimum(base, one - a)
                    preds.append((one - lam) * base + lam * cand)
                agg_neg = preds[0]
                for t in preds[1:]:
                    agg_neg = torch.minimum(agg_neg, t)
                out[..., h] = agg_neg
            else:
                out[..., h] = base

        return out

    # ------------------------- Main forward -------------------------
    def forward(self, x, *args, **kwargs):
        # During super().__init__ dry-run, lambda_head may not exist yet
        if not hasattr(self, 'lambda_head'):
            return super().forward(x, *args, **kwargs)

        # Step counter (training only)
        if self.training:
            self._global_step += 1

        original_output = super().forward(x, *args, **kwargs)

        # Take the last feature map, generate dynamic λ
        dynamic_lambdas = None
        if isinstance(original_output, list):
            fm = original_output[-1]                # [B,C,H,W]
            dynamic_lambdas = self.lambda_head(fm)  # [B,K]
        elif isinstance(original_output, tuple):
            feats = original_output[1]
            if isinstance(feats, list) and len(feats) > 0:
                fm = feats[-1]
                dynamic_lambdas = self.lambda_head(fm)  # [B,K]

        # Linear warm-up + global scale + per-slot scale
        if dynamic_lambdas is not None:
            warm = 1.0
            if self.training and self.lambda_warmup_batches > 0:
                warm = min(1.0, self._global_step / float(self.lambda_warmup_batches))
            dyn_scale = warm * self.lambda_global_scale
            dynamic_lambdas = dynamic_lambdas * dyn_scale
            # Per-slot extra scaling (register_buffer)
            dynamic_lambdas = dynamic_lambdas * self.slot_scale.view(1, -1).to(dynamic_lambdas.dtype)
            # Safe range
            dynamic_lambdas = torch.nan_to_num(dynamic_lambdas, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 0.999)

        def apply_rules(pred_tensor, lambdas_for_batch):
            if lambdas_for_batch is None:
                return pred_tensor
            box_part, cls_part, info = self._split_pred_tensor(pred_tensor, self.nc)
            if info['type'] == 'unknown':
                return pred_tensor

            B = cls_part.shape[0]
            if info['type'] == '4d':
                # Training: logits -> prob -> rules -> logit
                p = torch.sigmoid(cls_part)

                # Sparse diagnostics: before rules
                if self.training and (self._global_step % self._dbg_every == 0):
                    with torch.no_grad():
                        pm, pmin, pmax = p.mean().item(), p.min().item(), p.max().item()
                        lm = dynamic_lambdas.mean().item() if dynamic_lambdas is not None else -1
                        LOGGER.info(f"[rules] step={self._global_step} prob_before mean={pm:.4f} "
                                    f"min={pmin:.2e} max={pmax:.2e} lambda_mean={lm:.3f}")

                lamb = lambdas_for_batch.to(dtype=p.dtype, device=p.device).view(B, 1, 1, self.num_rules)
                p = self._apply_rules_prob(p, lamb, self.rules)
                # After-rule safeguard + print
                p = torch.nan_to_num(p, nan=0.0, posinf=1.0, neginf=0.0).clamp(1e-4, 1.0 - 1e-4)

                if self.training and (self._global_step % self._dbg_every == 0):
                    with torch.no_grad():
                        pm, pmin, pmax = p.mean().item(), p.min().item(), p.max().item()
                        LOGGER.info(f"[rules] step={self._global_step} prob_after  mean={pm:.4f} "
                                    f"min={pmin:.2e} max={pmax:.2e}")

                new_logits = torch.log(p) - torch.log(1.0 - p)
                return self._merge_pred_tensor(box_part, new_logits, info)

            elif info['type'] == '3d':
                # Inference/validation: should be probability. If range is abnormal, apply sigmoid correction.
                with torch.no_grad():
                    mn = float(cls_part.min()); mx = float(cls_part.max())
                if (mn < 0.0) or (mx > 1.0):
                    p = torch.sigmoid(cls_part)
                else:
                    p = cls_part
                lamb = lambdas_for_batch.to(dtype=p.dtype, device=p.device).view(B, 1, self.num_rules)
                p = self._apply_rules_prob(p, lamb, self.rules)
                p = torch.nan_to_num(p, nan=0.0, posinf=1.0, neginf=0.0).clamp(1e-6, 1.0 - 1e-6)
                return self._merge_pred_tensor(box_part, p, info)
            else:
                return pred_tensor

        # Dispatch: apply rules to tensors in output structure
        if isinstance(original_output, tuple):
            preds_part = original_output[0]
            if isinstance(preds_part, list):
                processed = [apply_rules(t, dynamic_lambdas) for t in preds_part]
            else:
                processed = apply_rules(preds_part, dynamic_lambdas)
            return (processed,) + original_output[1:]
        elif isinstance(original_output, list):
            return [apply_rules(t, dynamic_lambdas) for t in original_output]
        else:
            return apply_rules(original_output, dynamic_lambdas)

    # ------------------------- Loss -------------------------
    def init_criterion(self):
        return YOLOLoss(self, req_loss=self.req_loss)


# ===================================================================
# Trainer & Dataset interface (keep as-is)
# ===================================================================
class YOLOTrainer(DetectionTrainer):

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None, req_loss=False):
        super().__init__(cfg, overrides, _callbacks)
        self.req_loss = req_loss
    
    def build_dataset(self, img_path, mode='train', batch=None):
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        dataset = build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == 'val',
                                     stride=gs,
                                     lb_class_id=self.model.lb_class_id,
                                     lb_id_class=self.model.lb_id_class,
                                     lb_id_class_norm=self.model.lb_id_class_norm)
        self.model.lb_class_id = dataset.lb_class_id
        self.model.lb_id_class = dataset.lb_id_class
        self.model.lb_id_class_norm = dataset.lb_id_class_norm
        return dataset

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = YOLOModel(cfg, nc=self.data['nc'], verbose=verbose and RANK == -1, req_loss=self.req_loss)
        model.lb_class_id = {}
        model.lb_id_class = {}
        model.lb_id_class_norm = {}
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        if self.req_loss:
            self.loss_names = 'box_loss', 'cls_loss', 'dfl_loss', 'req_loss'
        else:
            self.loss_names = 'box_loss', 'cls_loss', 'dfl_loss'
        validator = YOLOValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))
        return validator


def build_yolo_dataset(cfg, img_path, batch, data, mode='train', rect=False, stride=32,
                       lb_class_id={}, lb_id_class={}, lb_id_class_norm={}):
    """Build YOLO Dataset"""
    return ROADYOLODataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == 'train',
        hyp=cfg,
        rect=cfg.rect or rect,
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=int(stride),
        pad=0.0 if mode == 'train' else 0.5,
        prefix=colorstr(f'{mode}: '),
        use_segments=cfg.task == 'segment',
        use_keypoints=cfg.task == 'pose',
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if mode == 'train' else 1.0,
        lb_class_id=lb_class_id,
        lb_id_class=lb_id_class,
        lb_id_class_norm=lb_id_class_norm)
#fixed version:
# File path: YOLO/trainer.py
import torch
import torch.nn as nn
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.tal import dist2bbox, make_anchors
from .loss import YOLOLoss
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.torch_utils import de_parallel
from ultralytics.utils.plotting import plot_labels
from ultralytics.utils import DEFAULT_CFG, RANK, colorstr, LOGGER, TQDM
from .dataset import ROADYOLODataset
from .validator import YOLOValidator
from copy import copy
import numpy as np


# ===================================================================
# YOLOModel: rule injection with fixed λ (multiple rules, single pass)
# ===================================================================
class YOLOModel(DetectionModel):

    def __init__(self, cfg='yolov8x.yaml', ch=3, nc=None, verbose=True,
                 req_loss=False, fixed_lambdas_default=None):
        self._custom_ready = False
        super().__init__(cfg, ch, nc, verbose)

        self.nc = nc
        self.req_loss = req_loss
        self.num_rule_passes = 1  # single pass

        # ---------------- Rule table (your 3-layer set of 13 rules merged into one list) ----------------
        # Notation:
        #   sign: +1 = use h[i]; -1 = use 1 - h[i]
        #   head_sign: +1 = positive head -> smax; -1 = negative head -> smin
        #
        # Class indices (ROAD-R):
        # 0:Ped 1:Car 2:Cyc 3:Mobike 4:MedVeh 5:LarVeh 6:Bus 7:EmVeh 8:TL 9:OthTL
        # 10:Red 11:Amber 12:Green 13:MovAway 14:MovTow 15:Mov 16:Brake 17:Stop
        # 18:IncatLft 19:IncatRht 20:HazLit 21:TurLft 22:TurRht 23:Ovtak 24:Wait2X
        # 25:XingFmLft 26:XingFmRht 27:Xing 28:PushObj 29:VehLane 30:OutgoLane
        # 31:OutgoCycLane 32:IncomLane 33:IncomCycLane 34:Pav 35:LftPav 36:RhtPav
        # 37:Jun 38:xing 39:BusStop 40:parking
        self.rules = [
            # ---------- Layer 1 (8 rules) ----------
            {"head_idx": 23, "head_sign": -1, "body": [(17, -1)], "lambda_slot": 0},          # n23 :- n17
            {"head_idx": 18, "head_sign": -1, "body": [(0,  +1)], "lambda_slot": 1},          # n18 :- 0
            {"head_idx": 12, "head_sign": -1, "body": [(10, +1)], "lambda_slot": 2},          # n12 :- 10
            {"head_idx": 28, "head_sign": -1, "body": [(0,  -1)], "lambda_slot": 3},          # n28 :- n0
            {"head_idx": 36, "head_sign": +1, "body": [(0,  +1), (35, -1)], "lambda_slot": 4},# 36 :- 0 n35
            {"head_idx": 20, "head_sign": -1, "body": [(32, -1)], "lambda_slot": 5},          # n20 :- n32
            {"head_idx": 28, "head_sign": -1, "body": [(1,  +1)], "lambda_slot": 6},          # n28 :- 1
            {"head_idx": 38, "head_sign": -1, "body": [(30, +1)], "lambda_slot": 7},          # n38 :- 30

            # ---------- Layer 2 (4 rules) ----------
            {"head_idx": 24, "head_sign": +1,
             "body": [(0,+1),(13,-1),(14,-1),(15,-1),(17,-1),(21,-1),(22,-1),(25,-1),(26,-1),(27,-1),(28,-1)],
             "lambda_slot": 8},                                                                # 24 :- 0 n13 ... n28
            {"head_idx": 8,  "head_sign": +1, "body": [(9,-1),(12,+1)], "lambda_slot": 9},     # 8 :- n9 12
            {"head_idx": 7,  "head_sign": -1, "body": [(28,+1)],   "lambda_slot": 10},         # n7 :- 28
            {"head_idx": 8,  "head_sign": +1, "body": [(9,-1),(10,+1)], "lambda_slot": 11},    # 8 :- n9 10

            # ---------- Layer 3 (1 rule) ----------
            {"head_idx": 4,  "head_sign": +1,
             "body": [(1,-1),(2,-1),(3,-1),(5,-1),(6,-1),(7,-1),(18,+1)],
             "lambda_slot": 12},                                                               # 4 :- n1 n2 n3 n5 n6 n7 18
        ]

        total_rules = len(self.rules)  # 13
        # Fixed λ (one λ per rule)
        if fixed_lambdas_default is None:
            fixed_lambdas_default = [0] * total_rules
        lamb = torch.tensor(fixed_lambdas_default, dtype=torch.float32)
        self.fixed_lambdas = nn.Parameter(lamb, requires_grad=False)

        self._custom_ready = True

    # ------------------------- Utilities -------------------------
    @staticmethod
    def _split_pred_tensor(pred_tensor, nc):
        info = {'type': None, 'layout': None, 'shape': tuple(pred_tensor.shape)}
        if pred_tensor.ndim == 4:
            b, c, h, w = pred_tensor.shape
            pt = pred_tensor.permute(0, 2, 3, 1)
            reg_max = 16
            box_ch = reg_max * 4
            box_part = pt[..., :box_ch]
            cls_part = pt[..., box_ch:]
            info.update(type='4d', layout='bhwc', reg_max=reg_max)
            return box_part, cls_part, info
        elif pred_tensor.ndim == 3:
            B, A, B_or_N = pred_tensor.shape
            if A == 4 + nc:
                pt = pred_tensor.transpose(1, 2).contiguous()
                return pt[..., :4], pt[..., 4:], {'type': '3d', 'layout': 'b_n_4c_channels_first'}
            elif B_or_N == 4 + nc:
                return pred_tensor[..., :4], pred_tensor[..., 4:], {'type': '3d', 'layout': 'b_n_4c_channels_last'}
            else:
                return None, None, {'type': 'unknown'}
        else:
            return None, None, {'type': 'unknown'}

    @staticmethod
    def _merge_pred_tensor(box_part, cls_part, info):
        if info['type'] == '4d':
            return torch.cat([box_part, cls_part], dim=-1).permute(0, 3, 1, 2).contiguous()
        elif info['type'] == '3d':
            recomb = torch.cat([box_part, cls_part], dim=-1)
            if info['layout'] == 'b_n_4c_channels_first':
                return recomb.transpose(1, 2).contiguous()
            else:
                return recomb
        else:
            return torch.cat([box_part, cls_part], dim=-1)

    # ------------------------- Rule logic (supports passing a rule set) -------------------------
    def _apply_rules_prob(self, cls_prob, lambdas, rules):
        src = cls_prob
        out = cls_prob.clone()
        one = torch.ones(1, dtype=src.dtype, device=src.device)

        buckets = {}
        for r in rules:
            head, head_sign, body, ls = r["head_idx"], r["head_sign"], r["body"], r["lambda_slot"]
            old = src[..., head]

            # body score = min over literals (literal is h or 1 - h)
            a = None
            for (idx, sign) in body:
                lit = src[..., idx] if sign == +1 else (one - src[..., idx])
                a = lit if a is None else torch.minimum(a, lit)

            # Candidate: positive head uses max(old, a); negative head uses min(old, 1 - a)
            cand = torch.maximum(old, a) if head_sign == +1 else torch.minimum(old, one - a)

            lam = lambdas[..., ls]  # λ for this rule
            pred_k = (one - lam) * old + lam * cand  # weighted form of smax/smin

            buckets.setdefault((head, head_sign), []).append(pred_k)

        # Aggregate multiple candidates for the same head: positive -> max, negative -> min
        for (head, head_sign), preds in buckets.items():
            agg = preds[0]
            for t in preds[1:]:
                agg = torch.maximum(agg, t) if head_sign == +1 else torch.minimum(agg, t)
            out[..., head] = agg

        return out

    # ------------------------- Main forward -------------------------
    def forward(self, x, *args, **kwargs):
        if not getattr(self, "_custom_ready", False):
            return DetectionModel.forward(self, x, *args, **kwargs)

        original_output = DetectionModel.forward(self, x, *args, **kwargs)
        lambdas = self.fixed_lambdas.unsqueeze(0)  # [1, total_rules]

        def apply_rules(pred_tensor, lambdas_for_batch):
            box_part, cls_part, info = self._split_pred_tensor(pred_tensor, self.nc)

            if info['type'] == '4d':
                B = cls_part.shape[0]
                p = torch.sigmoid(cls_part)
                lamb = lambdas_for_batch.expand(B, -1).view(B, 1, 1, -1)  # [B,1,1,R]
                for _ in range(self.num_rule_passes):
                    p = self._apply_rules_prob(p, lamb, self.rules)
                p = p.clamp(1e-6, 1 - 1e-6)
                new_logits = torch.log(p) - torch.log(1 - p)
                return self._merge_pred_tensor(box_part, new_logits, info)

            elif info['type'] == '3d':
                B = cls_part.shape[0]
                p = cls_part  # already probabilities
                lamb = lambdas_for_batch.expand(B, -1).view(B, 1, -1)     # [B,1,R]
                for _ in range(self.num_rule_passes):
                    p = self._apply_rules_prob(p, lamb, self.rules)
                return self._merge_pred_tensor(box_part, p, info)
            else:
                return pred_tensor

        if isinstance(original_output, tuple):
            preds_part = original_output[0]
            if isinstance(preds_part, list):
                processed = [apply_rules(t, lambdas) for t in preds_part]
            else:
                processed = apply_rules(preds_part, lambdas)
            return (processed,) + original_output[1:]
        elif isinstance(original_output, list):
            return [apply_rules(t, lambdas) for t in original_output]
        else:
            return apply_rules(original_output, lambdas)

    def init_criterion(self):
        return YOLOLoss(self, req_loss=self.req_loss)


# ===================================================================
# Trainer & Dataset interface (kept as-is)
# ===================================================================
class YOLOTrainer(DetectionTrainer):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None, req_loss=False):
        super().__init__(cfg, overrides, _callbacks)
        self.req_loss = req_loss
    
    def build_dataset(self, img_path, mode='train', batch=None):
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        dataset = build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == 'val',
                                     stride=gs,
                                     lb_class_id=self.model.lb_class_id,
                                     lb_id_class=self.model.lb_id_class,
                                     lb_id_class_norm=self.model.lb_id_class_norm)
        self.model.lb_class_id = dataset.lb_class_id
        self.model.lb_id_class = dataset.lb_id_class
        self.model.lb_id_class_norm = dataset.lb_id_class_norm
        return dataset

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = YOLOModel(cfg, nc=self.data['nc'], verbose=verbose and RANK == -1, req_loss=self.req_loss)
        model.lb_class_id = {}
        model.lb_id_class = {}
        model.lb_id_class_norm = {}
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        if self.req_loss:
            self.loss_names = 'box_loss', 'cls_loss', 'dfl_loss', 'req_loss'
        else:
            self.loss_names = 'box_loss', 'cls_loss', 'dfl_loss'
        return YOLOValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))


def build_yolo_dataset(cfg, img_path, batch, data, mode='train', rect=False, stride=32,
                       lb_class_id={}, lb_id_class={}, lb_id_class_norm={}):
    return ROADYOLODataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == 'train',
        hyp=cfg,
        rect=cfg.rect or rect,
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=int(stride),
        pad=0.0 if mode == 'train' else 0.5,
        prefix=colorstr(f'{mode}: '),
        use_segments=cfg.task == 'segment',
        use_keypoints=cfg.task == 'pose',
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if mode == 'train' else 1.0,
        lb_class_id=lb_class_id,
        lb_id_class=lb_id_class,
        lb_id_class_norm=lb_id_class_norm)
#choose to run the code
