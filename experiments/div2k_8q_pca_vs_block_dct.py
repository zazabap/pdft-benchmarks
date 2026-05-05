#!/usr/bin/env python3
"""DIV2K-8q PCA-vs-block-DCT benchmark — placeholder.

The paper requires a DIV2K-8q (m=n=8, 256×256) analog of the QuickDraw
experiment, including MERA on the unblocked variant. The implementation
is deferred to a follow-on spec; this stub exists so the file path is
reserved and importers fail loudly rather than silently.
"""

import sys


def main() -> int:
    print(
        "experiments/div2k_8q_pca_vs_block_dct.py is a placeholder. "
        "The DIV2K-8q PCA-vs-block-DCT experiment is deferred to a "
        "follow-on spec; see results/div2k_8q_pca_vs_block_dct/README.md.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
