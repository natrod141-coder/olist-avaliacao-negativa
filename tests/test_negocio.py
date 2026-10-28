import numpy as np
import pytest

from olist.negocio import limiar_por_custo, resultado_esperado, tabela_capacidade


def test_limiar_por_custo():
    # Contato custa 5, evitar uma negativa vale 50, ação funciona em 20% dos casos:
    # vale contatar quando p * 0.2 * 50 > 5, ou seja, p > 0.5.
    assert limiar_por_custo(5, 50, 0.2) == pytest.approx(0.5)


def test_limiar_rejeita_parametros_invalidos():
    with pytest.raises(ValueError):
        limiar_por_custo(5, 50, 0)


def test_tabela_capacidade_modelo_perfeito():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    prob = np.array([0.9, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    tabela = tabela_capacidade(y, prob, percentuais=(0.2,))
    linha = tabela.iloc[0]
    assert linha["precisão"] == 1.0
    assert linha["recall"] == 1.0
    assert linha["lift"] == 5.0


def test_resultado_esperado():
    y = np.array([1, 0, 1, 0])
    prob = np.array([0.9, 0.8, 0.2, 0.1])
    r = resultado_esperado(y, prob, limiar=0.5, custo_contato=10, valor_evitado=100, efetividade=0.5)
    assert r["pedidos contatados"] == 2
    assert r["negativas alcançadas"] == 1
    assert r["resultado (R$)"] == pytest.approx(1 * 0.5 * 100 - 2 * 10)
