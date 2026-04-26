"""pdft_benchmarks: experiment harness for the pdft library.

Public surface:
- run_experiment(dataset, m, n, bases, baselines, preset, ...) -> Result   (pipeline.py)
- get_preset(dataset, name) -> Preset                                       (presets.py)
- BASIS_FACTORIES, BASELINE_FACTORIES                                       (bases.py / baselines.py)

Pinned to pdft >= 0.2.0.
"""

__version__ = "0.1.0"
