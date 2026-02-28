from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt

@dataclass
class _CDIState:
    chapter: str = "01"
    fig_counter: int = 0

_STATE = _CDIState()

def cdi_notebook_init(chapter: str) -> None:
    _STATE.chapter = str(chapter).zfill(2)
    _STATE.fig_counter = 0
    Path("figures").mkdir(exist_ok=True)

def show_and_save_mpl(fig: Optional[plt.Figure] = None, dpi: int = 160) -> str:
    if fig is None:
        fig = plt.gcf()
    _STATE.fig_counter += 1
    out = Path("figures") / f"{_STATE.chapter}_{_STATE.fig_counter:03d}.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.show()
    return str(out)
