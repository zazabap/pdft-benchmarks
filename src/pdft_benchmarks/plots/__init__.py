"""Plot helpers for benchmark results.

Only the shared paper style lives here now: the Wong palette, `save_figure`
(PDF + SVG, no PNG), and `set_paper_rcparams`. The former
`plot_loss_trajectories` / `plot_rate_distortion` helpers were used solely by
the Julia-schema report generator, which is not part of the paper pipeline.
"""

from .style import WONG, save_figure, set_paper_rcparams

__all__ = ["WONG", "save_figure", "set_paper_rcparams"]
