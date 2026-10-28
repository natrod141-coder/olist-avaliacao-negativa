"""Carregamento dos dados brutos e construção da base de modelagem.

A base é montada em SQL (DuckDB), lendo os CSVs direto do disco, sem
precisar de um banco de dados instalado.
"""

from pathlib import Path

import duckdb
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
PASTA_RAW = RAIZ / "data" / "raw"
PASTA_PROCESSADA = RAIZ / "data" / "processed"
ARQUIVO_SQL = RAIZ / "sql" / "base_modelagem.sql"
ARQUIVO_BASE = PASTA_PROCESSADA / "base_modelagem.parquet"

# Nome da view no SQL -> nome do arquivo CSV do Kaggle.
TABELAS = {
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "order_reviews": "olist_order_reviews_dataset.csv",
    "order_payments": "olist_order_payments_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "products": "olist_products_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
}


def conectar(pasta_raw: Path = PASTA_RAW) -> duckdb.DuckDBPyConnection:
    """Abre uma conexão DuckDB em memória com uma view para cada CSV."""
    faltando = [arq for arq in TABELAS.values() if not (pasta_raw / arq).exists()]
    if faltando:
        raise FileNotFoundError(
            f"Arquivos ausentes em {pasta_raw}: {faltando}. "
            "Veja no README como baixar o dataset."
        )
    con = duckdb.connect()
    for view, arquivo in TABELAS.items():
        caminho = (pasta_raw / arquivo).as_posix()
        con.execute(
            f"CREATE VIEW {view} AS SELECT * FROM read_csv_auto('{caminho}', header = true)"
        )
    return con


def construir_base(pasta_raw: Path = PASTA_RAW, salvar: bool = True) -> pd.DataFrame:
    """Executa o SQL da base de modelagem e devolve um DataFrame."""
    con = conectar(pasta_raw)
    base = con.execute(ARQUIVO_SQL.read_text(encoding="utf-8")).df()
    con.close()
    if salvar:
        PASTA_PROCESSADA.mkdir(parents=True, exist_ok=True)
        base.to_parquet(ARQUIVO_BASE, index=False)
    return base


def carregar_base() -> pd.DataFrame:
    """Lê a base já processada; constrói se ainda não existir."""
    if ARQUIVO_BASE.exists():
        return pd.read_parquet(ARQUIVO_BASE)
    return construir_base()


if __name__ == "__main__":
    df = construir_base()
    print(f"Base salva em {ARQUIVO_BASE}: {df.shape[0]} linhas, {df.shape[1]} colunas")
