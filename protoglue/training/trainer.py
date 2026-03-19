"""Training loop for ProtoGlue pretraining and fine-tuning stages."""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path


def temp_schedule(epoch: int, anneal_epochs: int, t_start: float, t_end: float) -> float:
    """Cosine annealing temperature schedule.

    Parameters
    ----------
    epoch : int
        Current epoch.
    anneal_epochs : int
        Total annealing epochs.
    t_start, t_end : float
        Start and end temperatures.

    Returns
    -------
    float
        Current temperature.
    """
    if anneal_epochs <= 0:
        return float(t_end)
    e = min(max(epoch, 0), anneal_epochs)
    cos = 0.5 * (1.0 + np.cos(np.pi * e / float(anneal_epochs)))
    return float(t_end + (t_start - t_end) * cos)


def run_stage(
    stage: str,
    epochs: int,
    model: nn.Module,
    X1_t: torch.Tensor,
    X2_t: torch.Tensor,
    A_sp_t: list,
    A_f1_t: torch.Tensor,
    A_f2_t: torch.Tensor,
    dec_head: nn.Module = None,
    optimizer: torch.optim.Optimizer = None,
    scheduler=None,
    scaler=None,
    loss_fn: callable = None,
    warm_offset: int = 0,
    ckpt_tag: str = None,
    output_dir: Path = None,
    device: torch.device = None,
    WARMUP_EPOCHS: int = 30,
    PATIENCE: int = 150,
    GRAD_CLIP: float = 3.0,
    DEC_UPDATE_INTERVAL: int = 5,
    P_SMOOTH: float = 0.0,
    COLLAPSE_CHECK_INTERVAL: int = 10,
    MIN_CLUSTER_FRAC: float = 0.01,
    SEED: int = 2022,
    TEMP_ANNEAL_EPOCHS: int = 200,
    TEMP_START: float = 1.0,
    TEMP_END: float = 0.25,
) -> list:
    """Run a training stage (pretraining or fine-tuning).

    Parameters
    ----------
    stage : str
        Either ``"pretrain"`` or ``"finetune"``.
    epochs : int
        Maximum number of epochs.
    model : nn.Module
        ProtoGlue model.
    X1_t, X2_t : Tensor
        Input feature tensors.
    A_sp_t : list of sparse Tensor
        Multi-scale spatial adjacency matrices.
    A_f1_t, A_f2_t : sparse Tensor
        Feature-space adjacency matrices.
    dec_head : nn.Module, optional
        DEC clustering head (required for fine-tuning).
    optimizer : Optimizer
        PyTorch optimizer.
    scheduler : LR scheduler, optional
        Learning rate scheduler.
    scaler : GradScaler, optional
        AMP gradient scaler.
    loss_fn : callable
        Loss function with signature ``(out, x1, x2, A_sp, p_target, warm, loss_weighter)``.
    warm_offset : int
        Epoch offset for warmup scheduling (used in fine-tuning).
    ckpt_tag : str, optional
        Tag for checkpoint filenames.
    output_dir : Path, optional
        Directory for saving checkpoints.
    device : torch.device, optional
        Computing device.
    WARMUP_EPOCHS, PATIENCE, GRAD_CLIP, etc.
        Training hyperparameters.

    Returns
    -------
    list of dict
        Training logs with loss components per epoch.
    """
    stage_name = ckpt_tag if ckpt_tag is not None else stage

    model.train()
    logs = []
    best = float("inf")
    bad = 0
    p_target = None

    for ep in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        epoch_abs = ep + warm_offset if stage == "finetune" else ep
        warm = min(1.0, epoch_abs / max(1, WARMUP_EPOCHS)) if stage == "finetune" else 0.0

        # Temperature annealing for inter-modality gate
        if stage == "finetune" and hasattr(model.inter, "set_temp"):
            T = temp_schedule(epoch_abs, TEMP_ANNEAL_EPOCHS, TEMP_START, TEMP_END)
            model.inter.set_temp(T)

        # Update DEC target distribution periodically
        if stage == "finetune" and (ep == 1 or epoch_abs % DEC_UPDATE_INTERVAL == 0):
            model.eval()
            with torch.no_grad():
                out_tmp = model(X1_t, X2_t, A_sp_t, A_f1_t, A_f2_t, dec_head=dec_head)
                p_target = dec_head.target_distribution(out_tmp["q"].detach())
                from protoglue.losses.clustering import smooth_target
                p_target = smooth_target(p_target, A_sp_t[0], P_SMOOTH)
            model.train()

        try:
            if scaler is not None:
                with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                    out = model(X1_t, X2_t, A_sp_t, A_f1_t, A_f2_t, dec_head=dec_head)
                    loss, parts = loss_fn(
                        out, X1_t, X2_t, A_sp_t[0],
                        p_target=p_target, warm=warm, loss_weighter=model.loss_weighter,
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + ([] if dec_head is None else list(dec_head.parameters())),
                    GRAD_CLIP,
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(X1_t, X2_t, A_sp_t, A_f1_t, A_f2_t, dec_head=dec_head)
                loss, parts = loss_fn(
                    out, X1_t, X2_t, A_sp_t[0],
                    p_target=p_target, warm=warm, loss_weighter=model.loss_weighter,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + ([] if dec_head is None else list(dec_head.parameters())),
                    GRAD_CLIP,
                )
                optimizer.step()

        except RuntimeError as e:
            # Sparse op + AMP incompatibility: fallback to fp32
            if scaler is not None and scaler.is_enabled():
                print("AMP fallback -> fp32 due to:", str(e).split("\n")[0])
                scaler = torch.amp.GradScaler("cuda", enabled=False)
                continue
            else:
                raise

        if scheduler is not None:
            scheduler.step(loss.detach().cpu())

        parts.update({
            "stage": stage_name,
            "epoch": int(epoch_abs),
            "epoch_local": int(ep),
            "lr": float(optimizer.param_groups[0]["lr"]),
        })
        logs.append(parts)

        # Cluster collapse detection (fine-tuning only)
        if (
            stage == "finetune"
            and dec_head is not None
            and (epoch_abs % max(1, COLLAPSE_CHECK_INTERVAL) == 0)
        ):
            try:
                from .callbacks import detect_collapse_from_q, reset_dead_prototypes
                bad_idx, frac, counts = detect_collapse_from_q(
                    out["q"], K=int(dec_head.mu.shape[0]), min_frac=MIN_CLUSTER_FRAC,
                )
                if len(bad_idx) > 0:
                    print(
                        f"[{stage_name}] collapse clusters={bad_idx.tolist()} "
                        f"frac={np.round(frac[bad_idx], 4)} -> resetting prototypes"
                    )
                    reset_dead_prototypes(dec_head, bad_idx, out["z"].detach(), seed=SEED + epoch_abs)
            except Exception:
                pass

        # Logging
        if ep == 1 or ep % 10 == 0:
            print(
                f"[{stage_name}] ep={epoch_abs:04d} loss={parts['loss']:.4f} "
                f"recon={parts['recon']:.4f} corr={parts['corr']:.4f} "
                f"cl={parts['cluster']:.4f} warm={parts['warm']:.2f}"
            )

        # Early stopping
        min_run = WARMUP_EPOCHS + 50 if stage == "finetune" else 30
        if parts["loss"] < best - 1e-6:
            best = parts["loss"]
            bad = 0
            if output_dir is not None:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "dec_head": None if dec_head is None else dec_head.state_dict(),
                    },
                    output_dir / f"best_{stage_name}.pt",
                )
        else:
            if ep > min_run:
                bad += 1
                if bad >= PATIENCE:
                    print(f"[{stage_name}] early stop at ep={epoch_abs}")
                    break

    return logs
