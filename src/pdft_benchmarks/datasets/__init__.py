"""Dataset loaders.

Each module exposes a `load_<name>` function with its own kwargs. The
dispatcher here is a thin re-export hub for the common case where the
caller has only the string name.
"""

from .div2k import load_div2k
from .quickdraw import load_quickdraw
from .tuberlin import load_tuberlin

_LOADERS = {
    "div2k": load_div2k,
    "quickdraw": load_quickdraw,
    "tuberlin": load_tuberlin,
}


def load(name: str, **kwargs):
    """Dispatch to load_<name>(**kwargs). Raises KeyError on unknown name."""
    if name not in _LOADERS:
        raise KeyError(f"unknown dataset {name!r}; choices: {sorted(_LOADERS)}")
    return _LOADERS[name](**kwargs)


__all__ = ["load", "load_div2k", "load_quickdraw", "load_tuberlin"]
