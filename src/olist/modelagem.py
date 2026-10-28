"""Variáveis, divisão temporal, pipelines dos modelos e métricas."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

ALVO = "avaliacao_negativa"
DATA_CORTE_TESTE = "2018-06-01"
SEMENTE = 42

# Variáveis numéricas com cauda longa: aplicamos log na regressão logística.
NUMERICAS_ASSIMETRICAS = [
    "horas_ate_aprovacao",
    "valor_produtos",
    "valor_frete",
    "razao_frete_produtos",
    "peso_total_g",
    "volume_total_cm3",
    "media_tamanho_descricao",
    "distancia_km",
    "vendedor_n_avaliacoes_previas",
]

NUMERICAS = [
    "prazo_prometido_dias",
    "dias_ate_postagem",
    "entregue_antes_pesquisa",
    "dias_atraso_observado",
    "dias_ate_entrega",
    "n_itens",
    "n_produtos",
    "n_vendedores",
    "media_fotos",
    "parcelas",
    "n_pagamentos",
    "usou_voucher",
    "mesmo_estado",
    "dia_semana_compra",
    "hora_compra",
    "vendedor_taxa_negativa_previa",
    *NUMERICAS_ASSIMETRICAS,
]

CATEGORICAS = ["categoria", "tipo_pagamento", "uf_cliente", "uf_vendedor"]

VARIAVEIS = NUMERICAS + CATEGORICAS


def preparar_tipos(df: pd.DataFrame) -> pd.DataFrame:
    """Converte as colunas categóricas para o tipo 'category' do pandas."""
    df = df.copy()
    for col in CATEGORICAS:
        df[col] = df[col].astype("category")
    return df


def dividir_treino_teste(df: pd.DataFrame, data_corte: str = DATA_CORTE_TESTE):
    """Divisão temporal: treino antes da data de corte, teste depois.

    Nunca embaralhamos: o modelo é treinado no passado e avaliado no futuro,
    como acontece em produção.
    """
    df = df.sort_values("data_pesquisa").reset_index(drop=True)
    treino = df[df["data_pesquisa"] < data_corte].reset_index(drop=True)
    teste = df[df["data_pesquisa"] >= data_corte].reset_index(drop=True)
    return treino, teste


def validacao_temporal(n_splits: int = 4) -> TimeSeriesSplit:
    """Validação cruzada em que cada fold de validação é posterior ao treino."""
    return TimeSeriesSplit(n_splits=n_splits)


# ----------------------------------------------------------------------
# Pipelines
# ----------------------------------------------------------------------

def pipeline_regressao_logistica(C: float = 1.0) -> Pipeline:
    """Regressão logística: precisa de imputação, escala e one-hot."""
    log_transform = FunctionTransformer(np.log1p, feature_names_out="one-to-one")
    numericas_simples = [c for c in NUMERICAS if c not in NUMERICAS_ASSIMETRICAS]

    preprocess = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputar", SimpleImputer(strategy="median", add_indicator=True)),
                        ("escalar", StandardScaler()),
                    ]
                ),
                numericas_simples,
            ),
            (
                "num_log",
                Pipeline(
                    [
                        ("imputar", SimpleImputer(strategy="median", add_indicator=True)),
                        ("log", log_transform),
                        ("escalar", StandardScaler()),
                    ]
                ),
                NUMERICAS_ASSIMETRICAS,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=100),
                CATEGORICAS,
            ),
        ]
    )
    return Pipeline(
        [
            ("preprocess", preprocess),
            ("modelo", LogisticRegression(C=C, max_iter=2000)),
        ]
    )


def pipeline_random_forest() -> Pipeline:
    """Random Forest: não precisa de escala; categorias viram códigos inteiros."""
    preprocess = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), NUMERICAS),
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                CATEGORICAS,
            ),
        ]
    )
    return Pipeline(
        [
            ("preprocess", preprocess),
            (
                "modelo",
                RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=20,
                    max_features="sqrt",
                    n_jobs=-1,
                    random_state=SEMENTE,
                ),
            ),
        ]
    )


def modelo_gradient_boosting(**params) -> HistGradientBoostingClassifier:
    """Gradient Boosting do scikit-learn.

    Lida sozinho com valores nulos e com colunas do tipo 'category', então
    não precisa de pré-processamento.
    """
    padrao = dict(
        learning_rate=0.05,
        max_iter=400,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        l2_regularization=1.0,
        categorical_features="from_dtype",
        early_stopping=False,
        random_state=SEMENTE,
    )
    padrao.update(params)
    return HistGradientBoostingClassifier(**padrao)


# ----------------------------------------------------------------------
# Métricas
# ----------------------------------------------------------------------

@dataclass
class Metricas:
    pr_auc: float
    roc_auc: float
    brier: float
    taxa_base: float

    def como_dict(self) -> dict:
        return {
            "PR-AUC": round(self.pr_auc, 4),
            "ROC-AUC": round(self.roc_auc, 4),
            "Brier": round(self.brier, 4),
            "Taxa base": round(self.taxa_base, 4),
        }


def avaliar(y_true, prob) -> Metricas:
    """PR-AUC é a métrica principal: foca na classe rara (avaliação negativa).

    A PR-AUC de um modelo aleatório é igual à taxa base, então ela deve ser
    lida em comparação com a taxa base.
    """
    y_true = np.asarray(y_true)
    prob = np.asarray(prob)
    return Metricas(
        pr_auc=average_precision_score(y_true, prob),
        roc_auc=roc_auc_score(y_true, prob),
        brier=brier_score_loss(y_true, np.clip(prob, 0, 1)),
        taxa_base=float(y_true.mean()),
    )


def validar_cruzado(modelo_fabrica, X: pd.DataFrame, y: pd.Series, n_splits: int = 4) -> pd.DataFrame:
    """Treina e avalia o modelo em cada fold temporal.

    `modelo_fabrica` é uma função que devolve um modelo novo, para que cada
    fold comece do zero.
    """
    resultados = []
    for fold, (idx_tr, idx_va) in enumerate(validacao_temporal(n_splits).split(X), start=1):
        modelo = modelo_fabrica()
        modelo.fit(X.iloc[idx_tr], y.iloc[idx_tr])
        prob = modelo.predict_proba(X.iloc[idx_va])[:, 1]
        m = avaliar(y.iloc[idx_va], prob)
        resultados.append({"fold": fold, "n_treino": len(idx_tr), "n_validacao": len(idx_va), **m.como_dict()})
    return pd.DataFrame(resultados)
