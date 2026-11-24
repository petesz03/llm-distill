import gzip
import hashlib
import urllib.request
from pathlib import Path
from typing import Literal

import numpy as np
import torch

# <workspace>/src/llm_distill/dataset/mnist.py -> <workspace>/data
DATA_ROOT = Path(__file__).resolve().parents[3] / "data"
MNIST_DIR = DATA_ROOT / "mnist"

MIRRORS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
]

FILES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_valid(path: Path, md5: str) -> bool:
    return path.is_file() and _md5(path) == md5


def _download(filename: str, md5: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    errors = []
    for mirror in MIRRORS:
        url = mirror + filename
        try:
            print(f"Downloading {url}")
            urllib.request.urlretrieve(url, tmp)
        except OSError as e:
            errors.append(f"{url}: {e}")
            continue
        if _md5(tmp) == md5:
            tmp.replace(dest)
            return
        errors.append(f"{url}: MD5 mismatch")
    tmp.unlink(missing_ok=True)
    raise RuntimeError(f"Failed to download {filename}:\n  " + "\n  ".join(errors))


def download_mnist(root: Path | str = MNIST_DIR) -> Path:
    """Download the four MNIST files into ``root``, skipping any already present.

    Returns the directory containing the files.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for filename, md5 in FILES.items():
        dest = root / filename
        if _is_valid(dest, md5):
            continue
        _download(filename, md5, dest)
    return root


def _read_idx(path: Path, ndim: int) -> np.ndarray:
    """Read a gzipped IDX file of unsigned bytes with ``ndim`` dimensions."""
    with gzip.open(path, "rb") as f:
        data = f.read()
    magic = int.from_bytes(data[0:4], "big")
    if magic != 0x0800 + ndim:  # 0x08 = unsigned byte
        raise ValueError(f"{path}: unexpected IDX magic number {magic:#x}")
    shape = tuple(int.from_bytes(data[4 + 4 * i : 8 + 4 * i], "big") for i in range(ndim))
    return np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * ndim).reshape(shape)


def load_mnist(
    split: Literal["train", "test"] = "train",
    root: Path | str = MNIST_DIR,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load an MNIST split, downloading the files first if needed.

    Returns:
        images: ``uint8`` tensor of shape ``(N, 28, 28)``, raw pixel values in [0, 255].
        labels: ``int64`` tensor of shape ``(N,)``, digit classes 0-9.

    ``N`` is 60,000 for ``"train"`` and 10,000 for ``"test"``.
    """
    prefix = {"train": "train", "test": "t10k"}[split]
    root = download_mnist(root)
    images = _read_idx(root / f"{prefix}-images-idx3-ubyte.gz", ndim=3)
    labels = _read_idx(root / f"{prefix}-labels-idx1-ubyte.gz", ndim=1)
    return torch.from_numpy(images.copy()), torch.from_numpy(labels.astype(np.int64))


if __name__ == "__main__":
    for split in ("train", "test"):
        x, y = load_mnist(split)
        print(f"{split}: images {tuple(x.shape)} {x.dtype}, labels {tuple(y.shape)} {y.dtype}")
