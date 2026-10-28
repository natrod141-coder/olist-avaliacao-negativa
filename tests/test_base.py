"""Testes da base de modelagem, com foco em vazamento de dados.

Rodar com:  pytest -q
"""

import numpy as np
import pandas as pd
import pytest

from olist.dados import PASTA_RAW, TABELAS, construir_base
from olist.modelagem import ALVO, VARIAVEIS

pytestmark = pytest.mark.skipif(
    not all((PASTA_RAW / arq).exists() for arq in TABELAS.values()),
    reason="dados brutos não encontrados em data/raw",
)


@pytest.fixture(scope="module")
def base():
    return construir_base(salvar=False)


@pytest.fixture(scope="module")
def brutos():
    ler = lambda nome: pd.read_csv(PASTA_RAW / TABELAS[nome])
    return {nome: ler(nome) for nome in ["orders", "order_reviews", "order_items"]}


def test_uma_linha_por_pedido(base):
    assert base["order_id"].is_unique


def test_alvo_binario_e_taxa_plausivel(base):
    assert set(base[ALVO].unique()) == {0, 1}
    assert 0.10 < base[ALVO].mean() < 0.16


def test_nenhuma_variavel_usa_informacao_da_avaliacao(base):
    """Nota, texto e data de resposta não podem entrar como variáveis."""
    proibidas = ("review", "score", "resposta", "comment", "nota")
    assert not [v for v in VARIAVEIS if any(p in v for p in proibidas)]


def test_coerencia_entrega_ate_a_pesquisa(base):
    """Tempo de entrega só existe para pedidos entregues antes da pesquisa."""
    entregue = base["entregue_antes_pesquisa"] == 1
    assert base.loc[entregue, "dias_ate_entrega"].notna().all()
    assert base.loc[~entregue, "dias_ate_entrega"].isna().all()


def test_tempos_nao_negativos(base):
    for col in ["dias_ate_postagem", "dias_ate_entrega", "prazo_prometido_dias"]:
        assert (base[col].dropna() >= 0).all(), col


def test_historico_do_vendedor_usa_apenas_o_passado(base, brutos):
    """Recalcula o histórico em pandas, de forma independente do SQL.

    Para cada pedido de uma amostra, conta as avaliações de outros pedidos do
    mesmo vendedor respondidas ANTES do envio da pesquisa do pedido atual.
    """
    orders, reviews, items = brutos["orders"], brutos["order_reviews"], brutos["order_items"]

    for col in ["review_creation_date", "review_answer_timestamp"]:
        reviews[col] = pd.to_datetime(reviews[col])
    primeira = (
        reviews.sort_values(["review_creation_date", "review_answer_timestamp"])
        .drop_duplicates("order_id")
    )
    principal = (
        items.sort_values(["order_id", "price", "order_item_id"], ascending=[True, False, True])
        .drop_duplicates("order_id")[["order_id", "seller_id"]]
    )
    entregues = orders[
        (orders["order_status"] == "delivered") & orders["order_delivered_customer_date"].notna()
    ][["order_id"]]
    pedidos = entregues.merge(primeira, on="order_id").merge(principal, on="order_id")

    amostra = base.sample(300, random_state=0)
    for _, linha in amostra.iterrows():
        mesmo_vendedor = pedidos[pedidos["seller_id"] == linha["vendedor_principal"]]
        esperado = (mesmo_vendedor["review_answer_timestamp"] < linha["data_pesquisa"]).sum()
        assert linha["vendedor_n_avaliacoes_previas"] == esperado, linha["order_id"]


def test_taxa_do_vendedor_suavizada(base):
    taxa = base["vendedor_taxa_negativa_previa"]
    # Nula apenas no começo da base, quando ainda não havia histórico global.
    assert taxa.isna().mean() < 0.01
    assert base.loc[taxa.isna(), "data_pesquisa"].max() < pd.Timestamp("2017-03-01")
    assert ((taxa.dropna() > 0) & (taxa.dropna() < 1)).all()
    # Vendedor sem histórico recebe a taxa global, nunca algo extremo.
    sem_historico = base["vendedor_n_avaliacoes_previas"] == 0
    assert taxa[sem_historico].dropna().between(0.05, 0.25).all()
