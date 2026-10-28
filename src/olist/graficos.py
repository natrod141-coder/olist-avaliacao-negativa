"""Estilo único para todos os gráficos do projeto."""

from pathlib import Path

import matplotlib.pyplot as plt

from olist.dados import RAIZ

PASTA_FIGURAS = RAIZ / "reports" / "figures"

# Paleta categórica validada para daltonismo (ordem fixa: azul, laranja, verde-água).
AZUL = "#2a78d6"
LARANJA = "#eb6834"
VERDE = "#1baf7a"
CINZA = "#8a8984"
TEXTO = "#0b0b0b"
TEXTO_SECUNDARIO = "#52514e"
FUNDO = "#fcfcfb"


def aplicar_estilo() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": FUNDO,
            "axes.facecolor": FUNDO,
            "savefig.facecolor": FUNDO,
            "axes.edgecolor": "#d9d8d4",
            "axes.labelcolor": TEXTO_SECUNDARIO,
            "axes.titlecolor": TEXTO,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#ecebe7",
            "grid.linewidth": 0.8,
            "xtick.color": TEXTO_SECUNDARIO,
            "ytick.color": TEXTO_SECUNDARIO,
            "lines.linewidth": 2,
            "legend.frameon": False,
            "figure.dpi": 110,
            "font.size": 10,
        }
    )


def salvar(fig, nome: str) -> Path:
    PASTA_FIGURAS.mkdir(parents=True, exist_ok=True)
    caminho = PASTA_FIGURAS / f"{nome}.png"
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    return caminho
