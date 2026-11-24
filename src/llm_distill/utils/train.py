import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

EvalStrategy = Literal["none", "end", "epoch", "batch"]

BestMetric = Literal["train_loss", "train_acc", "test_loss", "test_acc"]
_BEST_METRIC_KEYS = {
    "train_loss": "loss",
    "train_acc": "accuracy",
    "test_loss": "loss",
    "test_acc": "accuracy",
}


@dataclass
class TrainConfig:
    """Options for :func:`train`. Any field can also be passed as a keyword override."""

    epochs: int = 10
    batch_size: int = 128
    shuffle: bool = True
    drop_last: bool = False
    num_workers: int = 0

    optimizer: Literal["sgd", "adam", "adamw"] = "adam"  # ignored if train(optim=...)
    lr: float = 1e-3
    momentum: float = 0.9
    nesterov: bool = False
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0
    scheduler: Literal["cosine", "linear", "step"] | None = None  # stepped per batch
    warmup_steps: int = 0
    step_size_epochs: int = 1
    gamma: float = 0.1  # "step" decay factor per step_size_epochs
    grad_clip: float | None = None

    eval_strategy: EvalStrategy = "epoch"  # "batch" evaluates every eval_every_n_batches
    eval_every_n_batches: int = 100
    eval_batch_size: int = 1024

    device: str | None = None  # None picks CUDA when available
    bf16: bool = False  # bfloat16 autocast, CUDA only
    seed: int | None = 0
    log_every_n_batches: int = 100  # 0 disables per-step records
    verbose: bool = True


@dataclass
class History:
    """Records from :func:`train`: running loss per logged step, per-epoch summaries,
    evaluation results and the best saved checkpoint (if ``save_best_model`` was set)."""

    steps: list[dict[str, Any]] = field(default_factory=list)
    epochs: list[dict[str, Any]] = field(default_factory=list)
    eval: list[dict[str, Any]] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    best: dict[str, Any] | None = None


def _resolve_device(device: str | None) -> torch.device:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def _to_loader(
    data: Dataset | DataLoader,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    num_workers: int,
    generator: torch.Generator | None,
    pin_memory: bool,
) -> DataLoader:
    if isinstance(data, DataLoader):
        return data
    return DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        generator=generator,
        pin_memory=pin_memory,
    )


def _accuracy_counts(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor | None:
    """Number of correct argmax predictions, or ``None`` if targets aren't class indices."""
    if targets.dtype in (torch.int64, torch.int32) and logits.shape[:-1] == targets.shape:
        return (logits.argmax(-1) == targets).sum()
    return None


def save_checkpoint(model: nn.Module, path: str | Path, **metadata: Any) -> Path:
    """Save ``model``'s state dict (on CPU) plus ``metadata`` to ``path``, creating parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"model_state_dict": state, **metadata}, path)
    return path


def build_optimizer(params, cfg: TrainConfig) -> torch.optim.Optimizer:
    """Create the optimizer named in ``cfg``."""
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(
            params,
            lr=cfg.lr,
            momentum=cfg.momentum,
            nesterov=cfg.nesterov,
            weight_decay=cfg.weight_decay,
        )
    if cfg.optimizer == "adam":
        return torch.optim.Adam(params, lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay)
    raise ValueError(f"unknown optimizer {cfg.optimizer!r}")


def build_scheduler(
    optimizer: torch.optim.Optimizer, cfg: TrainConfig, steps_per_epoch: int
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Create a per-step LR scheduler from ``cfg``, or ``None``."""
    total = cfg.epochs * steps_per_epoch
    warmup = cfg.warmup_steps
    if cfg.scheduler is None and warmup == 0:
        return None

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        t = step - warmup
        span = max(1, total - warmup)
        if cfg.scheduler == "cosine":
            return 0.5 * (1 + math.cos(math.pi * min(t, span) / span))
        if cfg.scheduler == "linear":
            return max(0.0, 1 - t / span)
        if cfg.scheduler == "step":
            return cfg.gamma ** (t // (cfg.step_size_epochs * steps_per_epoch))
        if cfg.scheduler is None:
            return 1.0
        raise ValueError(f"unknown scheduler {cfg.scheduler!r}")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data: Dataset | DataLoader,
    loss_fn: LossFn = F.cross_entropy,
    *,
    batch_size: int = 1024,
    device: str | torch.device | None = None,
    bf16: bool = False,
) -> dict[str, float]:
    """Return ``{"loss", "n"}``, plus ``"accuracy"`` and ``"errors"`` for class targets."""
    device = _resolve_device(device) if not isinstance(device, torch.device) else device
    loader = _to_loader(data, batch_size, False, False, 0, None, device.type == "cuda")
    was_training = model.training
    model.eval()
    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros((), device=device, dtype=torch.long)
    has_acc = True
    n = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
            logits = model(inputs)
        logits = logits.float()
        bs = targets.shape[0]
        loss_sum += loss_fn(logits, targets) * bs
        c = _accuracy_counts(logits, targets)
        if c is None:
            has_acc = False
        else:
            correct += c
        n += bs
    model.train(was_training)

    out: dict[str, float] = {"loss": (loss_sum / max(n, 1)).item(), "n": n}
    if has_acc and n > 0:
        out["accuracy"] = correct.item() / n
        out["errors"] = n - correct.item()
    return out


def train(
    model: nn.Module,
    train_data: Dataset | DataLoader,
    test_data: Dataset | DataLoader | None = None,
    config: TrainConfig | None = None,
    *,
    loss_fn: LossFn = F.cross_entropy,
    optim: torch.optim.Optimizer | None = None,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    callback: Callable[[str, dict[str, Any], nn.Module], None] | None = None,
    save_best_model: str | Path | None = None,
    best_metric: BestMetric = "test_loss",
    **overrides: Any,
) -> History:
    """Train ``model`` on ``(inputs, targets)`` batches from ``train_data``.

    Args:
        train_data, test_data: Datasets or DataLoaders. ``config.eval_strategy`` sets when
            ``test_data`` is evaluated.
        loss_fn: ``loss_fn(logits, targets) -> scalar``.
        optim, lr_scheduler: Use these instead of building them from ``config``.
        callback: ``callback(event, record, model)`` for ``"step"``, ``"epoch"``, ``"eval"``.
        save_best_model: Checkpoint path. The model is saved after any epoch where
            ``best_metric`` improves. ``train_*`` metrics are the running averages over the
            epoch; ``test_*`` metrics evaluate ``test_data`` after every epoch.
        **overrides: ``TrainConfig`` fields, e.g. ``train(model, ds, epochs=3)``.
    """
    cfg = replace(config or TrainConfig(), **overrides)
    if cfg.eval_strategy not in ("none", "end", "epoch", "batch"):
        raise ValueError(f"unknown eval_strategy {cfg.eval_strategy!r}")
    if best_metric not in _BEST_METRIC_KEYS:
        raise ValueError(f"unknown best_metric {best_metric!r}")
    best_on_test = best_metric.startswith("test_")
    if save_best_model is not None and best_on_test and test_data is None:
        raise ValueError(f"best_metric {best_metric!r} requires test_data")
    device = _resolve_device(cfg.device)
    bf16 = cfg.bf16 and device.type == "cuda"
    do_eval = test_data is not None and cfg.eval_strategy != "none"

    generator = None
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        generator = torch.Generator().manual_seed(cfg.seed)

    loader = _to_loader(
        train_data,
        cfg.batch_size,
        cfg.shuffle,
        cfg.drop_last,
        cfg.num_workers,
        generator,
        device.type == "cuda",
    )
    steps_per_epoch = len(loader)

    model.to(device)
    optimizer = optim if optim is not None else build_optimizer(model.parameters(), cfg)
    scheduler = (
        lr_scheduler
        if lr_scheduler is not None
        else build_scheduler(optimizer, cfg, steps_per_epoch)
    )

    history = History(config=asdict(cfg))

    def log(msg: str) -> None:
        if cfg.verbose:
            print(msg)

    def emit(event: str, record: dict[str, Any]) -> None:
        getattr(history, {"step": "steps", "epoch": "epochs", "eval": "eval"}[event]).append(record)
        if callback is not None:
            callback(event, record, model)

    def eval_only() -> dict[str, float]:
        assert test_data is not None
        return evaluate(
            model, test_data, loss_fn, batch_size=cfg.eval_batch_size, device=device, bf16=bf16
        )

    def run_eval(step: int, epoch: int) -> dict[str, float]:
        res = eval_only()
        record = {"step": step, "epoch": epoch, **res}
        emit("eval", record)
        acc = f"  acc {res['accuracy']:.4f}  errors {res['errors']}" if "accuracy" in res else ""
        log(f"  [eval] epoch {epoch}  step {step}  loss {res['loss']:.4f}{acc}")
        return res

    def maybe_save_best(
        train_res: dict[str, Any], test_res: dict[str, float] | None, step: int, epoch: int
    ) -> None:
        key = _BEST_METRIC_KEYS[best_metric]
        res = test_res if best_on_test else train_res
        assert res is not None
        if key not in res:
            raise ValueError(f"best_metric {best_metric!r} needs {key!r}, got {list(res)}")
        value = res[key]
        if history.best is not None:
            prev = history.best["value"]
            if (value >= prev) if key == "loss" else (value <= prev):
                return
        assert save_best_model is not None
        path = save_checkpoint(
            model,
            save_best_model,
            epoch=epoch,
            step=step,
            metric=best_metric,
            value=value,
            train=train_res,
            eval=test_res,
        )
        history.best = {
            "epoch": epoch,
            "step": step,
            "metric": best_metric,
            "value": value,
            "path": str(path),
        }
        log(f"  [best] epoch {epoch}  {best_metric} {value:.4f}  saved to {path}")

    log(
        f"Training on {device} for {cfg.epochs} epochs x {steps_per_epoch} steps "
        f"(eval: {cfg.eval_strategy if test_data is not None else 'no test set'})"
    )

    step = 0
    model.train()
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.perf_counter()
        epoch_loss = torch.zeros((), device=device)
        epoch_correct = torch.zeros((), device=device, dtype=torch.long)
        epoch_has_acc = True
        epoch_n = 0
        window_loss = torch.zeros((), device=device)
        window_n = 0

        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
                logits = model(inputs)
            loss = loss_fn(logits.float(), targets)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            step += 1

            bs = targets.shape[0]
            detached = loss.detach() * bs
            epoch_loss += detached
            window_loss += detached
            epoch_n += bs
            window_n += bs
            c = _accuracy_counts(logits.detach(), targets)
            if c is None:
                epoch_has_acc = False
            else:
                epoch_correct += c

            if cfg.log_every_n_batches > 0 and step % cfg.log_every_n_batches == 0:
                record = {
                    "step": step,
                    "epoch": epoch,
                    "loss": (window_loss / window_n).item(),
                    "lr": optimizer.param_groups[0]["lr"],
                }
                emit("step", record)
                log(f"epoch {epoch}  step {step}  loss {record['loss']:.4f}  lr {record['lr']:.3g}")
                window_loss.zero_()
                window_n = 0

            if do_eval and cfg.eval_strategy == "batch" and step % cfg.eval_every_n_batches == 0:
                run_eval(step, epoch)

        record = {
            "epoch": epoch,
            "step": step,
            "loss": (epoch_loss / max(epoch_n, 1)).item(),
            "time": time.perf_counter() - t0,
        }
        if epoch_has_acc and epoch_n > 0:
            record["accuracy"] = epoch_correct.item() / epoch_n
        emit("epoch", record)
        acc = f"  acc {record['accuracy']:.4f}" if "accuracy" in record else ""
        log(f"epoch {epoch} done  train loss {record['loss']:.4f}{acc}  ({record['time']:.1f}s)")

        res = run_eval(step, epoch) if do_eval and cfg.eval_strategy == "epoch" else None
        if save_best_model is not None:
            if res is None and best_on_test:
                res = eval_only()
            maybe_save_best(record, res, step, epoch)

    if do_eval and cfg.eval_strategy == "end":
        run_eval(step, cfg.epochs)

    return history
