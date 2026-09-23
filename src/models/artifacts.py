"""Validate local model files before passing them to a binary loader."""
from pathlib import Path


def require_model_file(path: str | Path) -> Path:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"model file is missing: {path}")
    with path.open('rb') as stream:
        header = stream.read(128)
    if header.startswith(b'version https://git-lfs.github.com/spec/v1'):
        raise ValueError(
            f"{path.name} is a Git LFS pointer; run git lfs pull on the host and rebuild the image"
        )
    if not header:
        raise ValueError(f"model file is empty: {path.name}")
    return path
