"""Tradução das probabilidades do modelo em decisões de negócio."""

import numpy as np
import pandas as pd


def tabela_capacidade(y_true, prob, percentuais=(0.05, 0.10, 0.15, 0.20, 0.30)) -> pd.DataFrame:
    """Se o time só consegue contatar X% dos pedidos, quais ele deve escolher?

    Ordena os pedidos do maior para o menor risco e mede, para cada
    percentual contatado:
    - precisão: dos contatados, quantos iam mesmo avaliar mal;
    - recall: das avaliações negativas, quantas foram alcançadas;
    - lift: quantas vezes melhor que escolher pedidos ao acaso.
    """
    y_true = np.asarray(y_true)
    prob = np.asarray(prob)
    ordem = np.argsort(-prob, kind="stable")
    y_ordenado = y_true[ordem]
    total_negativas = y_true.sum()
    taxa_base = y_true.mean()

    linhas = []
    for p in percentuais:
        k = max(1, int(round(p * len(y_true))))
        acertos = y_ordenado[:k].sum()
        precisao = acertos / k
        linhas.append(
            {
                "% contatado": f"{p:.0%}",
                "pedidos contatados": k,
                "negativas encontradas": int(acertos),
                "precisão": round(precisao, 3),
                "recall": round(acertos / total_negativas, 3),
                "lift": round(precisao / taxa_base, 2),
                "limiar de probabilidade": round(float(prob[ordem][k - 1]), 3),
            }
        )
    return pd.DataFrame(linhas)


def limiar_por_custo(custo_contato: float, valor_evitado: float, efetividade: float) -> float:
    """Limiar de decisão que maximiza o resultado esperado.

    Contatar um pedido com probabilidade p de avaliação negativa gera, em
    média, p * efetividade * valor_evitado de benefício e custa custo_contato.
    Vale a pena contatar quando o benefício esperado supera o custo:

        p * efetividade * valor_evitado > custo_contato
        p > custo_contato / (efetividade * valor_evitado)

    Isso só faz sentido se as probabilidades estiverem calibradas.
    """
    if efetividade <= 0 or valor_evitado <= 0:
        raise ValueError("efetividade e valor_evitado precisam ser positivos")
    return custo_contato / (efetividade * valor_evitado)


def resultado_esperado(y_true, prob, limiar: float, custo_contato: float, valor_evitado: float, efetividade: float) -> dict:
    """Resultado de aplicar o limiar no conjunto de teste."""
    y_true = np.asarray(y_true)
    contatar = np.asarray(prob) >= limiar
    n_contatos = int(contatar.sum())
    negativas_alcancadas = int(y_true[contatar].sum())
    beneficio = negativas_alcancadas * efetividade * valor_evitado
    custo = n_contatos * custo_contato
    return {
        "limiar": round(limiar, 3),
        "pedidos contatados": n_contatos,
        "% contatado": round(n_contatos / len(y_true), 3),
        "negativas alcançadas": negativas_alcancadas,
        "negativas evitadas (estimativa)": round(negativas_alcancadas * efetividade, 1),
        "custo (R$)": round(custo, 2),
        "benefício (R$)": round(beneficio, 2),
        "resultado (R$)": round(beneficio - custo, 2),
    }
