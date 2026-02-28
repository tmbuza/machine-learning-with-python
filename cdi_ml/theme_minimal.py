# ===========================================
# cdi_viz/theme.py
# Ultra-clean minimal CDI visualization helpers
# ===========================================
"""
CDI Visualization Helpers (Minimal)

Design goals:
- One shared chapter counter for all plot types
- Deterministic figure names: figures/{chapter}_{counter:03d}.png
- Minimal, reliable "finish" for Matplotlib/Seaborn (title/subtitle/legend/grid)
- Minimal Plotly layout polish (centered title + optional subtitle)
- Simple show-and-save helpers for Matplotlib, Plotly, and Plotnine

Notes:
- This file is intentionally small: it prioritizes teaching clarity over styling complexity.
- Plotly static export requires kaleido: `pip install -U kaleido`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Any

import matplotlib.pyplot as plt

# Optional: inline image embedding (works well in Quarto/Jupyter)
try:  # pragma: no cover
    from IPython.display import Markdown, display  # type: ignore
except Exception:  # pragma: no cover
    Markdown = None
    display = None


# ----------------------------
# Palette (optional, minimal)
# ----------------------------
CDI_PALETTE = {
    "ink": "#374151",
    "muted": "#6B7280",
    "title": "#036281",
    "grid": "#E5E7EB",
}


# ----------------------------
# Shared chapter + counter
# ----------------------------
_STATE = {"chapter": "01", "counter": 0}


def cdi_notebook_init(*, chapter: str) -> None:
    """
    Initialize per-chapter state and ensure figures/ exists.

    Usage:
        from cdi_viz.theme import cdi_notebook_init
        cdi_notebook_init(chapter="02")
    """
    _STATE["chapter"] = str(chapter).zfill(2)
    _STATE["counter"] = 0
    Path("figures").mkdir(parents=True, exist_ok=True)


def _next_path(*, folder: str = "figures", ext: str = "png") -> str:
    _STATE["counter"] += 1
    Path(folder).mkdir(parents=True, exist_ok=True)
    name = f"{_STATE['chapter']}_{_STATE['counter']:03d}.{ext}"
    return str(Path(folder) / name)


def _embed(path: str) -> None:
    """Embed an image in notebook/Quarto if IPython is available."""
    if display is None or Markdown is None:
        print(path)
        return
    display(Markdown(f"![]({path})"))


# ----------------------------
# Matplotlib / Seaborn finishing
# ----------------------------
def cdi_mpl_finish(
    fig: plt.Figure,
    ax: plt.Axes,
    *,
    title: str,
    subtitle: Optional[str] = None,
    legend: bool = False,
    legend_title: Optional[str] = None,
    legend_ncol: int = 2,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Minimal finishing touches for Matplotlib/Seaborn axes:
    - Figure title (regular weight, centered)
    - Subtitle (axes title, bold, centered)
    - Y-grid only (light)
    - Optional legend above the panel (clean, frame-less)

    This aims to mimic the clean readability of ggplot2 without heavy theming.
    """
    fig.suptitle(title, y=1.02, ha="center")

    if subtitle:
        ax.set_title(subtitle, pad=10, fontweight="bold", loc="center")
    else:
        ax.set_title("", pad=10)

    ax.grid(True, axis="y", linewidth=0.4, alpha=0.3)
    ax.grid(False, axis="x")

    if legend:
        ax.legend(
            title=legend_title,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=legend_ncol,
            frameon=False,
        )

    fig.tight_layout()
    return fig, ax


def show_and_save_mpl(
    fig: Optional[plt.Figure] = None,
    *,
    dpi: int = 300,
    folder: str = "figures",
    emit_markdown: bool = True,
    close: bool = True,
) -> str:
    """
    Save Matplotlib figure to figures/{chapter}_{counter}.png and optionally embed it.
    """
    if fig is None:
        fig = plt.gcf()

    out = _next_path(folder=folder, ext="png")
    fig.savefig(out, dpi=dpi, bbox_inches="tight")

    if emit_markdown:
        _embed(out)

    if close:
        plt.close(fig)

    return out


# ----------------------------
# Plotly: minimal layout polish + show + save
# ----------------------------
def set_cdi_plotly_layout(
    fig: Any,
    *,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    title_x: float = 0.5,
    legend_orientation: str = "h",
    legend_y: float = -0.25,
    legend_x: float = 0.0,
    margin: Optional[dict] = None,
) -> Any:
    """
    Minimal Plotly polish:
    - Centered title
    - Optional subtitle (rendered inside title via HTML)
    - Consistent legend position
    """
    if margin is None:
        margin = dict(l=40, r=20, t=85, b=55)

    if title is not None and subtitle is not None:
        full_title = (
            f"{title}<br><span style='font-size:0.85em; color:{CDI_PALETTE['muted']}'>"
            f"{subtitle}</span>"
        )
        fig.update_layout(title=full_title)
    elif title is not None:
        fig.update_layout(title=title)

    fig.update_layout(
        title_x=title_x,
        margin=margin,
        legend=dict(orientation=legend_orientation, y=legend_y, x=legend_x),
    )
    return fig


def show_and_save_plotly(
    fig: Any,
    *,
    folder: str = "figures",
    scale: int = 2,
    show: bool = False,
    emit_markdown: bool = True,
) -> str:
    """
    Save Plotly figure to figures/{chapter}_{counter}.png (requires kaleido).

    - show=False (default): best for Quarto rendering (avoids duplicate outputs)
    - show=True: interactive view + saved PNG (use during development)
    """
    out = _next_path(folder=folder, ext="png")

    if show:
        fig.show()

    try:
        fig.write_image(out, scale=scale)
    except Exception as e:  # pragma: no cover
        msg = (
            "Plotly PNG export failed. Install kaleido:\n"
            "  pip install -U kaleido\n"
            f"Reason: {e}"
        )
        raise RuntimeError(msg) from e

    if emit_markdown:
        _embed(out)

    return out


# ----------------------------
# Plotnine: show + save
# ----------------------------
def show_and_save_plotnine(
    p: Any,
    *,
    folder: str = "figures",
    dpi: int = 300,
    emit_markdown: bool = True,
    **kwargs: Any,
) -> str:
    """
    Save Plotnine ggplot object to figures/{chapter}_{counter}.png and embed it.
    kwargs passed to p.save (e.g., width=, height=, units=).
    """
    out = _next_path(folder=folder, ext="png")
    p.save(out, dpi=dpi, verbose=False, **kwargs)

    if emit_markdown:
        _embed(out)

    return out


__all__ = [
    "CDI_PALETTE",
    "cdi_notebook_init",
    "cdi_mpl_finish",
    "show_and_save_mpl",
    "set_cdi_plotly_layout",
    "show_and_save_plotly",
    "show_and_save_plotnine",
]
