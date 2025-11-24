import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
except Exception:
    pass


import pprint
import yaml

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from evals.latent_similarity.utils import (
    init_models,
    load_pretrained_model,
    )
from src.datasets.data_manager import init_data
from src.utils.distributed import init_distributed
from src.utils.logging import get_logger
from src.utils.tensors import unnormalize_tensor
from tqdm import tqdm

pp = pprint.PrettyPrinter(indent=4)

_GLOBAL_SEED = 123
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logger = get_logger(__name__)

from src.models.utils.quantization import  VectorQuantization
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_codebook_similarity(quant_module: VectorQuantization):
    # Get the number of embeddings (Codebook size N)
    N = quant_module.codebook.num_embeddings
    
    # Generate all indices from 0 to N-1
    all_indices = torch.arange(N, dtype=torch.long, device=quant_module.codebook.weight.device)
    
    with torch.no_grad():
        # Decode all Code vectors (Codebook Lookup)
        all_codes = quant_module.decode(all_indices)
        
        # L2 normalization of all Code vectors
        all_codes_norm = F.normalize(all_codes, p=2, dim=1)
        
        # Calculate the Cosine Similarity Matrix (S = A @ A.T)
        similarity_matrix = torch.matmul(all_codes_norm, all_codes_norm.T)

    # Create a mask to exclude the diagonal elements (similarity of a vector with itself)
    mask = ~torch.eye(N, dtype=torch.bool, device=similarity_matrix.device)
    non_diag_similarities = similarity_matrix[mask]
    
    # Calculate statistics
    mean_similarity = non_diag_similarities.mean().item()
    std_similarity = non_diag_similarities.std().item()
    min_similarity = non_diag_similarities.min().item()
    max_similarity = non_diag_similarities.max().item()

    logger.info("--- Statistical Results ---")
    logger.info(f"Average Cosine Similarity : {mean_similarity:.4f}")
    logger.info(f"Standard Deviation of Similarity : {std_similarity:.4f}")
    logger.info(f"Minimum Similarity (Least Similar Code Vectors): {min_similarity:.4f}")
    logger.info(f"Maximum Similarity (Most Similar Code Vectors): {max_similarity:.4f}")

    if max_similarity > 0.999:
        logger.warning("Highly similar or duplicate Code vectors exist! (Max Sim > 0.999)")
        # Find the two indices with the highest similarity (Optional)
        # Subtract the diagonal and multiply by 2 (or any large number) to ensure diagonal is not selected
        max_val, max_idx = torch.max(similarity_matrix - torch.eye(N, device=similarity_matrix.device) * 2, dim=0) 
        max_idx_flat = torch.argmax(max_val)
        idx_i = max_idx[max_idx_flat].item()
        idx_j = max_idx_flat.item()
        logger.warning(f"  E.g., Code {idx_i} and Code {idx_j} have a similarity of {max_similarity:.4f}")
        
    if N <= 100:  # Only visualize for small Codebooks
        # Ensure matplotlib and seaborn are imported if running this block
        # import matplotlib.pyplot as plt
        # import seaborn as sns
        
        plt.figure(figsize=(10, 8))
        # Convert PyTorch Tensor to numpy array
        matrix_np = similarity_matrix.cpu().numpy()
        
        sns.heatmap(
            matrix_np, 
            cmap='coolwarm',  # Use 'coolwarm' colormap to highlight similarity and dissimilarity
            vmin=-1, 
            vmax=1, 
            annot=False,  # Set to True to display numbers on cells
            fmt=".2f",
            square=True,
            cbar_kws={'label': 'Cosine Similarity'}
        )
        plt.title(f'Codebook (N={N}) Cosine Similarity Matrix')
        plt.xlabel('Code Index (j)')
        plt.ylabel('Code Index (i)')
        plt.show()
    else:
        logger.warning("(Codebook size is too large; skipping heatmap visualization.)")

def build_masks(B: int, t: int, p: int, device):
    mask = torch.ones((t, p), dtype=torch.int32).to(device)
    mask[-1, :] = 0
    mask = mask.flatten()
    mask_p = torch.argwhere(mask == 0).reshape(1, -1).expand(B, -1)
    mask_e = torch.nonzero(mask).reshape(1, -1).expand(B, -1)
    return [mask_e], [mask_p]


def load_yaml_config(yaml_path):
    try:
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        logger.error(f"Config file not found: {yaml_path}")
        return None
    except yaml.YAMLError as exc:
        logger.error(f"Error parsing YAML file {yaml_path}: {exc}")
        return None

def main(args_eval, resume_preempt=False):
    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #
    eval_log = args_eval.get("eval_log", "")

    config_path = os.path.join(eval_log, "params-pretrain.yaml")
    logger.info(f"Loading pretrain config from: {config_path}")
    pretrain_cfg = load_yaml_config(config_path)

    if pretrain_cfg is not None:
        args_eval["meta"] = pretrain_cfg.get("meta", {})
        args_eval["data"] = pretrain_cfg.get("data", {})
        args_eval["model"] = pretrain_cfg.get("model", {})
        logger.info("Updated args_eval with parameters from params-pretrain.yaml")
    else:
        logger.warning("Could not load pretrain config. Using parameters from args_eval as is.")
        
    cfgs_meta = pretrain_cfg.get("meta", {})
    cfgs_data = pretrain_cfg.get("data", {})
    cfgs_model = pretrain_cfg.get("model", {})

    num_frames = cfgs_data.get("num_frames")

    tubelet_size = cfgs_data.get("tubelet_size")
    patch_size = cfgs_data.get("patch_size")
    crop_size = cfgs_data.get("crop_size", 224)
    num_patches_per_frame = (crop_size // patch_size) ** 2


    # model pieces
    video_model_params = {
        "uniform_power": cfgs_model.get("uniform_power", True),
        "patch_size": patch_size,
        "num_frames": num_frames,
        "tubelet_size": tubelet_size,
        "model_name": cfgs_model.get("model_name"),
        "crop_size": crop_size,
        "pred_depth": cfgs_model.get("pred_depth"),
        "pred_embed_dim": cfgs_model.get("pred_embed_dim"),
        "use_sdpa": cfgs_meta.get("use_sdpa", False),
    }

    la_enc_params = {
        "num_heads": cfgs_model.get("latent_action_num_heads", 8),
        "d_codebook": cfgs_model.get("dims_aciton_codebook", 32),
        "n_codebook": cfgs_model.get("number_aciton_codebook", 1),
        "vq_bias": cfgs_model.get("vq_bias", True),
        "vq_commit_weight": cfgs_model.get("vq_commit_weight", 0.25),
        "vq_entropy_weight": cfgs_model.get("vq_entropy_weight", 0.1),
        "vq_diversity_weight": cfgs_model.get("vq_diversity_weight", 1.0),
        "use_sdpa": cfgs_meta.get("use_sdpa", False),
        "num_patches_per_frame": num_patches_per_frame,
    }

    num_frames = cfgs_data.get("num_frames")
    tubelet_size = cfgs_data.get("tubelet_size")
    crop_size = cfgs_data.get("crop_size", 224)
    patch_size = cfgs_data.get("patch_size")
    folder = args_eval.get("logging", {}).get("folder", None)
    

    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #

    # dtype / mixed precision
    which_dtype = cfgs_meta.get("dtype", "float32")
    logger.info(f"{which_dtype=}")
    # if str(which_dtype).lower() == "bfloat16":
    #     dtype = torch.bfloat16
    #     mixed_precision = True
    # elif str(which_dtype).lower() in ("float16", "fp16", "half"):
    #     dtype = torch.float16
    #     mixed_precision = True
    # else:
    #     dtype = torch.float32
    #     mixed_precision = False

    # -- set device
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    # -- init torch distributed backend
    world_size, rank = init_distributed()
    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")

    # -- init model
    encoder, target_encoder, ac_predictor, latent_action_enc = init_models(
        device=device,
        video_model_params=video_model_params,
        la_enc_params=la_enc_params,
    )
    model_dict = {
        "encoder": encoder,
        "target_encoder": target_encoder,
        "ac_predictor": ac_predictor,
        "latent_action_enc": latent_action_enc,
    }

    # -- freeze encoder
    for k, m in model_dict.items():
        for p in m.parameters():
            p.requires_grad = False
        logger.info(f"Freeze the {k}")

    # -- load pretrainig model checkpoint
    ac_jepa_pth = os.path.join(eval_log,"ac_jepa-latest.pth.tar")
    encoder, target_encoder, ac_predictor, latent_action_enc = load_pretrained_model(
        model_path=ac_jepa_pth,
        encoder=encoder,
        target_encoder=target_encoder,
        ac_predictor=ac_predictor,
        latent_action_enc=latent_action_enc,
        trainable=False,
        use_ddp=False,
    )

    # -- Evaluation
    results_dir = os.path.join(folder if folder is not None else ".", "eval_outputs")
    os.makedirs(results_dir, exist_ok=True)

    encoder.eval()
    ac_predictor.eval()
    latent_action_enc.eval()

    analyze_codebook_similarity(latent_action_enc.quant)
