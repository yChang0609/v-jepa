# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import copy
import os
from glob import glob

import src.models.ac_predictor as vit_ac_pred
import src.models.vision_transformer as video_vit
import torch
import torch.nn as nn

from einops import rearrange
from src.models.latent_action import LatentActionEncoder
from src.models.utils.multimask import MultiMaskWrapper, PredictorMultiMaskWrapper
from src.utils.logging import get_logger

logger = get_logger(__name__)



def _strip_module_prefix(sd):
    return {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}


def _load_component(checkpoint, name, model, use_ddp):
    """
    Load a single submodule state dict from a composite checkpoint.

    Args:
        checkpoint (dict): A dictionary containing submodule state dicts
                           under keys like 'encoder', 'target_encoder', etc.
        name (str): The submodule name to load, e.g. 'encoder'.
        model (nn.Module): The instantiated model/submodule to load into.
        use_ddp (bool): If True, assume checkpoint keys are already aligned
                        with DDP ("module.") prefixes and DO NOT strip them.
                        If False, strip leading "module." from keys.

    Returns:
        nn.Module: The same model instance after attempting to load weights.
    """
    if model is not None and name in checkpoint:
        try:
            # NOTE:
            # - When training with DDP, checkpoints often store parameter
            #   names prefixed by "module.".
            # - If `use_ddp` is True, we keep keys as-is.
            # - If `use_ddp` is False, we strip the "module." prefix to match
            #   non-DDP module param names.
            ckpt = checkpoint[name] if use_ddp else _strip_module_prefix(checkpoint[name])
            msg = model.load_state_dict(ckpt)
            logger.info(f"Loaded {name} with msg: {msg}")
        except Exception as e:
            logger.warning(f"Failed to load {name}: {e}")
    else:
        logger.warning(f'No "{name}" found in checkpoint.')
    return model


def _set_trainability(module: nn.Module, trainable: bool, eval_when_frozen: bool = True):
    """
    Turn gradients on/off for a module in a unified way.
    - When trainable=False, set requires_grad(False) for all params and optionally call eval()
      to stop BatchNorm running stats from updating.
    - When trainable=True, set requires_grad(True) for all params and call train().
    """
    if module is None:
        return
    for p in module.parameters():
        p.requires_grad_(trainable)
    if trainable:
        module.train()
    else:
        if eval_when_frozen:
            module.eval()


def load_pretrained_model(
    model_path,
    encoder,
    target_encoder,
    ac_predictor,
    latent_action_enc,
    trainable: bool = True,
    use_ddp: bool = True,
):
    """
    Load pretrained weights for all modules and apply a unified gradient on/off switch.

    Args:
        model_path (dict): Paths to checkpoints.
            Required keys:
                - 'ac_jepa_pth': str
                    Checkpoint containing keys:
                        'encoder', 'target_encoder', 'ac_predictor', 'latent_action_encoder'
        encoder, target_encoder, ac_predictor, latent_action_enc (nn.Module): Modules to load.
        trainable (bool): Unified switch. If False, all modules will have requires_grad(False)
                          and be set to eval() to freeze BatchNorm running stats.
        use_ddp (bool): Keep 'module.' prefixes if True; otherwise strip them.

    Returns:
        tuple: (encoder, target_encoder, ac_predictor, latent_action_enc)
    """

    # ---- 1) Load AC-JEPA-side modules ----
    print(model_path)
    try:
        ac_ckpt = torch.load(model_path, map_location=torch.device("cpu"))
    except Exception as e:
        logger.info(f'Exception when loading AC-JEPA checkpoint from "{model_path}": {e}')

    if ac_ckpt is not None:
        try:
            for name, module in [
                ("encoder", encoder),
                ("target_encoder", target_encoder),
                ("ac_predictor", ac_predictor),
                ("latent_action_encoder", latent_action_enc),
            ]:
                _ = _load_component(ac_ckpt, name, module, use_ddp)
        except Exception as e:
            logger.info(f"Failed to load AC-JEPA modules from checkpoint: {e}")
    else:
        logger.warning(
            "AC-JEPA checkpoint not loaded; encoder/target_encoder/ac_predictor/latent_action_enc remain as-initialized."
        )

    # ---- 3) Apply unified trainability to ALL modules ----
    for m in (encoder, target_encoder, ac_predictor, latent_action_enc):
        _set_trainability(m, trainable=trainable, eval_when_frozen=True)

    return encoder, target_encoder, ac_predictor, latent_action_enc


def init_models(
    device: torch.device,
    video_model_params: dict,
    la_enc_params: dict,
):
    """
    Initialize encoder, predictor, latent action encoder modules.

    Args:
        device (torch.device): Torch device to place the models on.
        video_model_params (dict): Parameters for initializing the video model.
            Required keys:
                - patch_size (int)
                - num_frames (int)
                - tubelet_size (int)
                - model_name (str)
                - uniform_power (bool)
                - use_sdpa (bool)
                - use_mask_tokens (bool)
                - num_mask_tokens (int)
                - zero_init_mask_tokens (bool)
                - crop_size (int)
                - pred_depth (int)
                - pred_embed_dim (int)
                - adapter_type (str)
                - action_dim (int)
        la_enc_params (dict): Parameters for initializing the latent action encoder.
            Required keys:
                - num_heads (int)
                - d_codebook (int)
                - n_codebook (int)
                - vq_bias (bool)
                - vq_commit_weight (float)
                - vq_entropy_weight (float)
                - vq_diversity_weight (float)

    Returns:
        tuple: (encoder, predictor, latent_action_enc)
    """

    # --------------------------
    # 1. Video Encoder & Predictor
    # --------------------------
    encoder, predictor = init_video_model(
        uniform_power=video_model_params["uniform_power"],
        device=device,
        patch_size=video_model_params["patch_size"],
        num_frames=video_model_params["num_frames"],
        tubelet_size=video_model_params["tubelet_size"],
        model_name=video_model_params["model_name"],
        crop_size=video_model_params["crop_size"],
        pred_depth=video_model_params["pred_depth"],
        pred_embed_dim=video_model_params["pred_embed_dim"],
        use_sdpa=video_model_params["use_sdpa"],
    )
    target_encoder = copy.deepcopy(encoder)

    # --------------------------
    # 2. Latent Action Encoder
    # --------------------------
    latent_action_enc = init_latent_action_encoder(
        device=device,
        input_dim=encoder.backbone.embed_dim,
        num_patches_per_frame=la_enc_params["num_patches_per_frame"],
        num_heads=la_enc_params["num_heads"],
        d_codebook=la_enc_params["d_codebook"],
        n_codebook=la_enc_params["n_codebook"],
        vq_bias=la_enc_params["vq_bias"],
        vq_commit_weight=la_enc_params["vq_commit_weight"],
        vq_entropy_weight=la_enc_params["vq_entropy_weight"],
        vq_diversity_weight=la_enc_params["vq_diversity_weight"],
        use_sdpa=la_enc_params["use_sdpa"],
    )

    return encoder, target_encoder, predictor, latent_action_enc


def init_video_model(
    device,
    patch_size=16,
    num_frames=16,
    tubelet_size=2,
    model_name="vit_base",
    crop_size=224,
    pred_depth=6,
    pred_embed_dim=384,
    uniform_power=False,
    use_sdpa=False,
) -> tuple[MultiMaskWrapper, PredictorMultiMaskWrapper]:
    encoder = video_vit.__dict__[model_name](
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        uniform_power=uniform_power,
        use_sdpa=use_sdpa,
    )
    encoder = MultiMaskWrapper(encoder)

    predictor = vit_ac_pred.__dict__["vit_ac_predictor"](
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        embed_dim=encoder.backbone.embed_dim,
        action_embed_dim=encoder.backbone.embed_dim,
        predictor_embed_dim=pred_embed_dim,
        depth=pred_depth,
        num_heads=encoder.backbone.num_heads,
        uniform_power=uniform_power,
        use_sdpa=use_sdpa,
        action_dim=encoder.backbone.embed_dim,
    )

    encoder.to(device)
    predictor.to(device)

    logger.info(encoder)
    logger.info(predictor)

    return encoder, predictor


def init_latent_action_encoder(
    device,
    input_dim: int,
    num_patches_per_frame: int,
    d_codebook: int,
    n_codebook: int,
    num_heads: int = 8,
    vq_bias: bool = True,
    vq_commit_weight: float = 0.25,
    vq_entropy_weight: float = 0.1,
    vq_diversity_weight: float = 1.0,
    use_sdpa: bool = True,
) -> LatentActionEncoder:
    la_enc = LatentActionEncoder(
        input_dim=input_dim,
        num_patches_per_frame=num_patches_per_frame,
        num_heads=num_heads,
        d_codebook=d_codebook,
        n_codebook=n_codebook,
        vq_bias=vq_bias,
        vq_commit_weight=vq_commit_weight,
        vq_entropy_weight=vq_entropy_weight,
        vq_diversity_weight=vq_diversity_weight,
        use_sdpa=use_sdpa,
    )

    la_enc = la_enc.to(device)
    logger.info(la_enc)

    return la_enc

