"""PDF -> HTML/CSS visual reconstruction pipeline (deterministic core, no ML/VLM).

Stage modules: extract.py, styles.py, layout.py, emit.py, render.py, diff.py.
See pipeline.run_skeleton for the current extract -> emit -> render -> diff
walking skeleton (styles.py/layout.py not implemented yet).
"""
