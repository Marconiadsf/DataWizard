from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, spearmanr, chi2_contingency
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor


from typing import Optional
from typing import List
import json

# ----------------------------------
# Schemas e funções para as ferramentas
# ----------------------------------

# Schema genérico para funções que aceitam coluna opcional
class ColunaInput(BaseModel):
    coluna: str = Field(
        default=None,
        description="Nome da coluna a ser analisada. Se não for fornecido, aplica em todas as colunas relevantes."
    )

# Schema vazio para funções sem argumentos
class SemArgs(BaseModel):
    pass


# Schema específico para ferramenta de histograma
class HistogramInput(BaseModel):
    coluna: str = Field(..., description="Nome da coluna numérica para gerar o histograma.")
    min_bins: Optional[int] = Field(None, description="Número mínimo de intervalos (bins).")
    max_bins: Optional[int] = Field(None, description="Número máximo de intervalos (bins).")
    default_bins: Optional[int] = Field(None, description="Número de intervalos padrão.")
    steps: Optional[int] = Field(None, description="Incremento do slider.")
    edge_color: Optional[str] = Field(None, description="Cor da borda das barras.")
    title: Optional[str] = Field(None, description="Título do gráfico.")
    xlabel: Optional[str] = Field(None, description="Rótulo do eixo X.")
    ylabel: Optional[str] = Field(None, description="Rótulo do eixo Y.")

# Schema específico para ferramenta de análise de outliers
class OutlierAnalyzerInput(BaseModel):
    coluna: str = Field(...,description="Nome da coluna numérica para análise de outliers via IQR.")
    title: Optional[str] = Field(None,description="Título opcional do boxplot (padrão: 'Boxplot de {coluna}').")
    ylabel: Optional[str] = Field(None,description="Rótulo opcional do eixo Y (padrão: nome da coluna).")
    iqr_multiplier: Optional[float] = Field(1.5,description="Multiplicador do IQR para definir limites de outliers (padrão 1.5).")
    incluir_resumo_estatistico: Optional[bool] = Field(True,description="Se verdadeiro, inclui Q1, Q3, IQR e proporção de outliers no texto.")
    incluir_recomendacoes: Optional[bool] = Field(True,description="Se verdadeiro, inclui recomendações (remover, transformar, investigar).")
    remover_outliers: Optional[bool] = Field(False,description="Se verdadeiro, remove os outliers antes de gerar o boxplot e compara estatísticas com a versão normal.")
    transformar_log: Optional[bool] = Field(False,description="Se verdadeiro, aplica transformação logarítmica (log1p) antes de gerar o boxplot e compara estatísticas com a versão normal.")
    winsorizar: Optional[bool] = Field(False,description="Se verdadeiro, aplica winsorização (limita valores aos limites do IQR) antes de gerar o boxplot e compara estatísticas com a versão normal.")

# Schema específico para ferramenta de scatterplot

class ScatterplotInput(BaseModel):
    coluna_x: str = Field(...,description="Nome da coluna numérica para o eixo X.")
    coluna_y: str = Field(...,description="Nome da coluna numérica para o eixo Y.")
    title: Optional[str] = Field(None,description="Título opcional do scatterplot (padrão: 'Relação entre {coluna_x} e {coluna_y}').")
    xlabel: Optional[str] = Field(None,description="Rótulo opcional do eixo X (padrão: nome da coluna X).")
    ylabel: Optional[str] = Field(None,description="Rótulo opcional do eixo Y (padrão: nome da coluna Y).")

# Schema específico para ferramenta de heatmap de clusters

class ClusterAnalyzerInput(BaseModel):
    max_clusters: Optional[int] = Field(6, description="Número máximo de clusters a testar (padrão: 6).")
    random_state: Optional[int] = Field(42, description="Semente aleatória para reprodutibilidade (padrão: 42).")
    min_cluster_size: Optional[int] = Field(0, description="Tamanho mínimo de observações por cluster (0 = não aplica).")
    scaling: Optional[str] = Field("standard", description="Método de normalização: 'standard' (z-score), 'minmax' ou 'none'.")

# Schema específico para ferramenta de tabela cruzada

class CrosstabAnalyzerInput(BaseModel):
    coluna_x: str = Field(..., description="Nome da coluna categórica para as linhas da tabela cruzada.")
    coluna_y: str = Field(..., description="Nome da coluna categórica para as colunas da tabela cruzada.")
    normalize: bool = Field(False, description="Se True, mostra proporções em vez de contagens absolutas.")
    output: str = Field("table", description="Formato de saída desejado: 'table' para tabela ou 'heatmap' para visualização gráfica.")
    max_table_dim: int = Field(10, description="Número máximo de linhas/colunas para exibir como tabela. Acima disso, força heatmap.")

# Schema específico para ferramenta de matriz de correlação

class CorrelationMatrixInput(BaseModel):
    method: str = Field("pearson", description="Método de correlação: 'pearson', 'spearman' ou 'kendall'.")
    top_n: int = Field(3, description="Número de pares mais correlacionados a destacar na interpretação.")

# Schema específico para ferramenta de análise de tendências temporais

class TemporalTrendsInput(BaseModel):
    coluna_tempo: str = Field(..., description="Nome da coluna temporal (datetime).")
    variaveis: List[str] = Field(..., description="Lista de variáveis numéricas a analisar.")
    freq: str = Field("D", description="Frequência de agregação: 'D' (diário), 'M' (mensal), 'Y' (anual).")
    agg: str = Field("mean", description="Função de agregação: 'mean', 'sum', etc.")

# Schema específico para ferramenta de importância de features

class FeatureImportanceInput(BaseModel):
    target: str = Field(..., description="Nome da variável alvo (dependente).")
    features: List[str] = Field(..., description="Lista de variáveis independentes (numéricas).")
    method: str = Field("linear", description="Método: 'linear' para regressão linear múltipla ou 'tree' para árvore de decisão.")
    max_depth: int = Field(4, description="Profundidade máxima da árvore (se method='tree').")
    random_state: int = Field(42, description="Semente aleatória para reprodutibilidade (se method='tree').")

# Schema específico para ferramenta de frequência de valores

class FrequenciaValoresInput(BaseModel):
    coluna: str = Field(..., description="Nome da coluna a ser analisada.")
    top_n: int = Field(5, description="Número de valores mais/menos frequentes a destacar.")

# -----------------------------------
# Funções das ferramentas
# -----------------------------------

def get_columns(*args, **kwargs):
    """Retorna a lista de colunas do DataFrame de amostra."""
    df = st.session_state.get("df_pandas_sample")
    if df is None:
        return "Nenhuma amostra disponível."
    return list(df.columns)

def get_summary(coluna: str = None, *args, **kwargs):
    """Resumo estatístico do DataFrame inteiro ou de uma coluna específica."""
    df = st.session_state.get("df_pandas_sample")
    if df is None:
        return "Nenhuma amostra disponível."
    if coluna and coluna in df.columns:
        return df[[coluna]].describe(include="all").to_dict()
    return df.describe(include="all").to_dict()

def get_central_tendency(coluna: str = None, *args, **kwargs):
    """Retorna média, mediana e moda de uma coluna numérica ou de todas as numéricas."""
    df = st.session_state.get("df_pandas_sample")
    if df is None:
        return "Nenhuma amostra disponível."

    medidas = {}

    # Se coluna específica foi pedida
    if coluna:
        if coluna not in df.columns:
            return f"A coluna '{coluna}' não existe."
        if not pd.api.types.is_numeric_dtype(df[coluna]):
            return f"A coluna '{coluna}' não é numérica. Use resumo_estatistico para analisá-la."
        moda = df[coluna].mode().iloc[0] if not df[coluna].mode().empty else None
        medidas[coluna] = {
            "media": df[coluna].mean(),
            "mediana": df[coluna].median(),
            "moda": moda
        }
        return medidas

    # Caso contrário, aplica em todas as numéricas
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            moda = df[col].mode().iloc[0] if not df[col].mode().empty else None
            medidas[col] = {
                "media": df[col].mean(),
                "mediana": df[col].median(),
                "moda": moda
            }
    return medidas

def get_variability(coluna: str = None, *args, **kwargs):
    """Retorna desvio padrão e variância de uma coluna numérica ou de todas as numéricas."""
    df = st.session_state.get("df_pandas_sample")
    if df is None:
        return "Nenhuma amostra disponível."

    variab = {}

    if coluna:
        if coluna not in df.columns:
            return f"A coluna '{coluna}' não existe."
        if not pd.api.types.is_numeric_dtype(df[coluna]):
            return f"A coluna '{coluna}' não é numérica. Use resumo_estatistico para analisá-la."
        variab[coluna] = {
            "desvio_padrao": df[coluna].std(),
            "variancia": df[coluna].var()
        }
        return variab

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            variab[col] = {
                "desvio_padrao": df[col].std(),
                "variancia": df[col].var()
            }
    return variab

def get_data_types(*args, **kwargs):
    """Retorna os tipos de dados de cada coluna, separando numéricas e categóricas."""
    df = st.session_state.get("df_pandas_sample")
    if df is None:
        return "Nenhuma amostra disponível."
    tipos = df.dtypes.apply(lambda x: str(x)).to_dict()
    numericos = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    categoricos = [col for col in df.columns if col not in numericos]
    return {
        "tipos": tipos,
        "numericos": numericos,
        "categoricos": categoricos
    }

def get_distributions(coluna: str = None, *args, **kwargs):
    """Retorna distribuições: estatísticas para numéricas, contagem para categóricas.
       Pode ser de uma coluna específica ou de todas."""
    df = st.session_state.get("df_pandas_sample")
    if df is None:
        return "Nenhuma amostra disponível."

    distrib = {}

    if coluna:
        if coluna not in df.columns:
            return f"A coluna '{coluna}' não existe."
        if pd.api.types.is_numeric_dtype(df[coluna]):
            distrib[coluna] = df[coluna].describe().to_dict()
        else:
            distrib[coluna] = df[coluna].value_counts().head(10).to_dict()
        return distrib

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            distrib[col] = df[col].describe().to_dict()
        else:
            distrib[col] = df[col].value_counts().head(10).to_dict()
    return distrib

def get_ranges(coluna: str = None, *args, **kwargs):
    """Retorna intervalo (mínimo e máximo) de uma coluna numérica ou de todas as numéricas."""
    df = st.session_state.get("df_pandas_sample")
    if df is None:
        return "Nenhuma amostra disponível."

    ranges = {}

    if coluna:
        if coluna not in df.columns:
            return f"A coluna '{coluna}' não existe."
        if not pd.api.types.is_numeric_dtype(df[coluna]):
            return f"A coluna '{coluna}' não é numérica. Use resumo_estatistico para analisá-la."
        ranges[coluna] = {
            "min": df[coluna].min(),
            "max": df[coluna].max()
        }
        return ranges

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            ranges[col] = {
                "min": df[col].min(),
                "max": df[col].max()
            }
    return ranges

# Funções auxiliares para sugestões de bins
def bins_sturges(data):
    n = len(data)
    return int(np.ceil(np.log2(n) + 1))

def bins_freedman_diaconis(data):
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    bin_width = 2 * iqr / (len(data) ** (1/3))
    if bin_width == 0:
        return 10  # fallback
    return int(np.ceil((data.max() - data.min()) / bin_width))


def bins_sturges(data):
    n = len(data)
    return int(np.ceil(np.log2(n) + 1))

def bins_freedman_diaconis(data):
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    bin_width = 2 * iqr / (len(data) ** (1/3))
    if bin_width == 0:
        return 10  # fallback
    return int(np.ceil((data.max() - data.min()) / bin_width))

# ---------------------------------
# Funções auxiliares para histograma
# ---------------------------------
def analisar_distribuicao(coluna: str, df: pd.DataFrame) -> dict:
    if coluna not in df.columns:
        return {"status": "error", "mensagem": f"A coluna '{coluna}' não existe."}

    if not pd.api.types.is_numeric_dtype(df[coluna]):
        return {"status": "error", "mensagem": f"A coluna '{coluna}' não é numérica."}

    serie = df[coluna].dropna()

    resultados = {
        "coluna": coluna,
        "n": len(serie),
        "min": float(serie.min()),
        "max": float(serie.max()),
        "media": float(serie.mean()),
        "mediana": float(serie.median()),
        "moda": float(serie.mode().iloc[0]) if not serie.mode().empty else None,
        "desvio_padrao": float(serie.std()),
        "variancia": float(serie.var()),
        "q1": float(serie.quantile(0.25)),
        "q3": float(serie.quantile(0.75)),
        "iqr": float(serie.quantile(0.75) - serie.quantile(0.25)),
        "assimetria": float(skew(serie)),
        "curtose": float(kurtosis(serie))
    }
    return resultados

def interpretar_distribuicao(stats: dict) -> str:
    if stats.get("status") == "error":
        return stats["mensagem"]

    media, mediana, skewness, curtose = stats["media"], stats["mediana"], stats["assimetria"], stats["curtose"]

    conclusoes = []

    # Simetria
    if abs(media - mediana) < 0.05 * stats["desvio_padrao"]:
        conclusoes.append("A distribuição é aproximadamente simétrica.")
    elif media > mediana:
        conclusoes.append("A distribuição é assimétrica à direita (cauda longa para valores altos).")
    else:
        conclusoes.append("A distribuição é assimétrica à esquerda (cauda longa para valores baixos).")

    # Dispersão
    if stats["desvio_padrao"] > 0.5 * (stats["max"] - stats["min"]):
        conclusoes.append("Os dados são bastante dispersos.")
    else:
        conclusoes.append("Os dados estão relativamente concentrados.")

    # Curtose
    if curtose > 0:
        conclusoes.append("A distribuição é mais pontuda que a normal (caudas pesadas).")
    else:
        conclusoes.append("A distribuição é mais achatada que a normal.")

    # Outliers potenciais
    limite_inf = stats["q1"] - 1.5 * stats["iqr"]
    limite_sup = stats["q3"] + 1.5 * stats["iqr"]
    if stats["min"] < limite_inf or stats["max"] > limite_sup:
        conclusoes.append("Há indícios de outliers nos extremos.")

    return " ".join(conclusoes)

# -----------------------------------
# Função principal para histograma
# -----------------------------------
def histogram_maker(
    coluna: str,
    min_bins: int = None,
    max_bins: int = None,
    default_bins: int = None,
    steps: int = None,
    edge_color: str = None,
    title: str = None,
    xlabel: str = None,
    ylabel: str = None
) -> str:
    """Gera um dicionário de informações para plotar um histograma pela interface Streamlit."""

    df = st.session_state.get("df_pandas_sample")
    if df is None:
        return json.dumps({"mensagem": "Nenhuma amostra disponível para análise.", "status": "error"})

    if coluna not in df.columns:
        return json.dumps({"mensagem": f"A coluna '{coluna}' não existe na amostra.", "status": "error"})

    if not pd.api.types.is_numeric_dtype(df[coluna]):
        return json.dumps({"mensagem": f"A coluna '{coluna}' não é numérica e não pode ser usada para histograma.", "status": "error"})
    
    data = df[coluna].dropna().values
    n = int(len(data))
    if n == 0:
        return json.dumps({"mensagem": f"A coluna '{coluna}' não possui dados válidos para histograma.", "status": "error"})

    # Sugestões automáticas
    sturges_raw = bins_sturges(data) if n > 1 else 1
    fd_raw = bins_freedman_diaconis(data) if n > 1 else 1

    try:
        sturges_bins = max(1, min(n, int(round(sturges_raw))))
    except Exception:
        sturges_bins = 1
    try:
        fd_bins = max(1, min(n, int(round(fd_raw))))
    except Exception:
        fd_bins = sturges_bins

    recomendacao_max = max(sturges_bins, fd_bins)
    recomendacao_min = min(sturges_bins, fd_bins)

    # Sanidade: não ter mais bins que pontos
    cap_max = max(1, n)

    # Defaults inteligentes
    min_bins = int(min_bins) if min_bins is not None else max(2 if n >= 2 else 1, recomendacao_min)
    max_bins = int(max_bins) if max_bins is not None else max(50, recomendacao_max)
    max_bins = min(max_bins, cap_max)

    # Ajuste de segurança: garante que min_bins não ultrapasse max_bins
    min_bins = min(min_bins, max_bins)

    # Default
    if default_bins is None:
        default_bins = sturges_bins if n < 10 else (fd_bins if fd_bins > sturges_bins else sturges_bins)
    default_bins = max(min_bins, min(default_bins, max_bins))

    # Passo dinâmico
    if steps is None:
        span = max_bins - min_bins
        steps = 1 if span <= 50 else max(1, int(round(span / 50)))
    else:
        steps = max(1, int(steps))

    edge_color = edge_color or "black"
    title = title or f"Histograma de {coluna}"
    xlabel = xlabel or coluna
    ylabel = ylabel or "Frequência"
    
    # Análise descritiva da coluna
    stats = analisar_distribuicao(coluna, df)
    interpretacao = interpretar_distribuicao(stats)

    mensagem = (
        f"Interpretação da distribuição da coluna '{coluna}':\n\n"
        f"{interpretacao}\n\n"
        f"📊 {title}\n\n"
        f"**Sugestões de intervalos:**\n"
        f"- Sturges: `{sturges_bins}`\n"
        f"- Freedman–Diaconis: `{fd_bins}`\n"
        f"(n={n}; slider: {min_bins}–{max_bins}, passo {steps}, padrão {default_bins})"
    )

    hist_inst = {
        "mensagem": mensagem,
        "acao": "histograma",
        "args": {
            "coluna": coluna,
            "min_bins": min_bins,
            "max_bins": max_bins,
            "default_bins": default_bins,
            "steps": steps,
            "edge_color": edge_color,
            "title": title,
            "xlabel": xlabel,
            "ylabel": ylabel
        },
        "status": "success"
    }
    return json.dumps(hist_inst)

# -----------------------------------
# Funções auxiliares de detecção e tratamento de outliers
# -----------------------------------

def calcular_outlier_stats(coluna: str, df: pd.DataFrame, iqr_multiplier: float = 1.5) -> dict:
    """Calcula estatísticas de outliers via regra do IQR com multiplicador configurável."""
    serie = df[coluna].dropna()
    q1 = serie.quantile(0.25)
    q3 = serie.quantile(0.75)
    iqr = q3 - q1
    limite_inf = q1 - iqr_multiplier * iqr
    limite_sup = q3 + iqr_multiplier * iqr

    outliers = serie[(serie < limite_inf) | (serie > limite_sup)]
    n_outliers = len(outliers)
    n_total = len(serie)

    return {
        "coluna": coluna,
        "n_total": n_total,
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "limite_inf": float(limite_inf),
        "limite_sup": float(limite_sup),
        "n_outliers": n_outliers,
        "prop_outliers": n_outliers / n_total if n_total > 0 else 0.0,
        "media": float(serie.mean()) if n_total > 0 else None,
        "mediana": float(serie.median()) if n_total > 0 else None,
        "desvio_padrao": float(serie.std()) if n_total > 0 else None
    }

def interpretar_outliers(stats: dict, incluir_resumo_estatistico: bool = True, incluir_recomendacoes: bool = True) -> str:
    """Gera interpretação textual dos outliers detectados."""
    if stats["n_total"] == 0:
        return "A coluna não possui dados válidos para análise de outliers."

    partes = []

    if stats["n_outliers"] == 0:
        partes.append(f"Não foram detectados valores atípicos na coluna '{stats['coluna']}'.")
    else:
        partes.append(
            f"Foram detectados {stats['n_outliers']} outliers "
            f"({stats['prop_outliers']:.1%} do total) na coluna '{stats['coluna']}'."
        )

    if incluir_resumo_estatistico:
        partes.append(
            f"(Q1={stats['q1']:.2f}, Q3={stats['q3']:.2f}, IQR={stats['iqr']:.2f}, "
            f"limite inferior={stats['limite_inf']:.2f}, limite superior={stats['limite_sup']:.2f})"
        )

    if stats["n_outliers"] > 0:
        if stats["prop_outliers"] > 0.2:
            partes.append("A proporção de outliers é alta, o que pode distorcer médias e análises.")
        elif stats["prop_outliers"] > 0.05:
            partes.append("Há uma quantidade moderada de outliers, que pode afetar algumas métricas.")
        else:
            partes.append("A quantidade de outliers é pequena e provavelmente não compromete a análise.")

    if incluir_recomendacoes and stats["n_outliers"] > 0:
        partes.append(
            "Esses valores podem ser investigados individualmente. "
            "Dependendo do contexto, podem ser removidos, transformados "
            "ou tratados como casos especiais."
        )

    return " ".join(partes)

# -----------------------------------
# Função principal para análise de outliers
# -----------------------------------

def outlier_analyzer(
    coluna: str,
    title: str = None,
    ylabel: str = None,
    iqr_multiplier: float = 1.5,
    incluir_resumo_estatistico: bool = True,
    incluir_recomendacoes: bool = True,
    remover_outliers: bool = False,
    transformar_log: bool = False,
    winsorizar: bool = False
) -> str:
    """Analisa outliers e, se solicitado, aplica transformações e compara com a situação normal."""

    df = st.session_state.get("df_pandas_sample")
    if df is None:
        return json.dumps({"mensagem": "Nenhuma amostra disponível para análise.", "status": "error"})

    if coluna not in df.columns:
        return json.dumps({"mensagem": f"A coluna '{coluna}' não existe na amostra.", "status": "error"})

    if not pd.api.types.is_numeric_dtype(df[coluna]):
        return json.dumps({"mensagem": f"A coluna '{coluna}' não é numérica e não pode ser usada em boxplot.", "status": "error"})

    serie_normal = df[coluna].dropna()

    # Estatísticas normais e interpretação
    stats_normal = calcular_outlier_stats(coluna, df, iqr_multiplier)
    interpretacao_normal = interpretar_outliers(stats_normal, incluir_resumo_estatistico, incluir_recomendacoes)

    # Determinar transformação escolhida (hierarquia)
    escolhas = [remover_outliers, winsorizar, transformar_log]
    if sum(escolhas) > 1:
        aviso = "⚠️ Mais de uma opção de tratamento foi passada. Aplicando hierarquia: remover_outliers > winsorizar > transformar_log."
        if remover_outliers:
            modo = "remover_outliers"
        elif winsorizar:
            modo = "winsorizar"
        else:
            modo = "transformar_log"
    elif remover_outliers:
        modo = "remover_outliers"; aviso = ""
    elif winsorizar:
        modo = "winsorizar"; aviso = ""
    elif transformar_log:
        modo = "transformar_log"; aviso = ""
    else:
        modo = "normal"; aviso = ""

    # Aplicar transformação
    if modo == "remover_outliers":
        q1, q3 = serie_normal.quantile([0.25, 0.75])
        iqr = q3 - q1
        limite_inf = q1 - iqr_multiplier * iqr
        limite_sup = q3 + iqr_multiplier * iqr
        serie_transf = serie_normal[(serie_normal >= limite_inf) & (serie_normal <= limite_sup)]
    elif modo == "winsorizar":
        q1, q3 = serie_normal.quantile([0.25, 0.75])
        iqr = q3 - q1
        limite_inf = q1 - iqr_multiplier * iqr
        limite_sup = q3 + iqr_multiplier * iqr
        serie_transf = serie_normal.clip(lower=limite_inf, upper=limite_sup)
    elif modo == "transformar_log":
        # Log seguro: somente valores >= 0. Para negativos, sugere-se investigar/ajustar antes.
        serie_transf = np.log1p(serie_normal[serie_normal >= 0])
    else:
        serie_transf = serie_normal

    # Estatísticas transformadas (para comparação)
    stats_transf = {
        "media": float(serie_transf.mean()) if len(serie_transf) > 0 else None,
        "mediana": float(serie_transf.median()) if len(serie_transf) > 0 else None,
        "desvio_padrao": float(serie_transf.std()) if len(serie_transf) > 0 else None
    }

    # Mensagem comparativa
    mensagem = (
        f"Análise de outliers da coluna '{coluna}':\n\n"
        f"{interpretacao_normal}\n\n"
    )
    if modo != "normal":
        mensagem += (
            f"Comparação com tratamento '{modo}':\n"
            f"- Média normal: {stats_normal['media']:.2f} → após: {stats_transf['media']:.2f}\n"
            f"- Mediana normal: {stats_normal['mediana']:.2f} → após: {stats_transf['mediana']:.2f}\n"
            f"- Desvio padrão normal: {stats_normal['desvio_padrao']:.2f} → após: {stats_transf['desvio_padrao']:.2f}\n\n"
            f"📊 Boxplot exibido corresponde à situação '{modo}'.\n"
        )
    else:
        mensagem += "📊 Boxplot exibido corresponde à situação normal.\n"

    if aviso:
        mensagem = aviso + "\n\n" + mensagem

    # Montagem do dicionário de instruções para o render
    title = title or f"Boxplot de {coluna}" + (f" ({modo})" if modo != "normal" else "")
    ylabel = ylabel or coluna

    box_inst = {
        "mensagem": mensagem,
        "acao": "boxplot",
        "args": {
            "coluna": coluna,
            "title": title,
            "ylabel": ylabel
        },
        "status": "success"
    }
    return json.dumps(box_inst)

# -----------------------------------
# Função auxiliar para scatterplot
# -----------------------------------

def calcular_correlacoes(df: pd.DataFrame, coluna_x: str, coluna_y: str) -> dict:
    """Calcula correlações de Pearson e Spearman entre duas colunas numéricas."""
    serie_x = df[coluna_x].dropna()
    serie_y = df[coluna_y].dropna()
    serie_x, serie_y = serie_x.align(serie_y, join="inner")

    pearson = serie_x.corr(serie_y, method="pearson")
    spearman, _ = spearmanr(serie_x, serie_y)

    return {
        "pearson": float(pearson),
        "spearman": float(spearman),
        "n": len(serie_x)
    }

def interpretar_correlacoes(corrs: dict, coluna_x: str, coluna_y: str) -> str:
    """Gera interpretação textual das correlações calculadas."""
    def qualifica(valor: float) -> str:
        v = abs(valor)
        if v < 0.2: return "muito fraca"
        elif v < 0.4: return "fraca"
        elif v < 0.6: return "moderada"
        elif v < 0.8: return "forte"
        else: return "muito forte"

    partes = []
    partes.append(
        f"A correlação de Pearson entre '{coluna_x}' e '{coluna_y}' é {corrs['pearson']:.2f}, "
        f"com base em {corrs['n']} observações."
    )
    partes.append(f"A correlação de Spearman é {corrs['spearman']:.2f}.")
    partes.append(
        f"Isso indica uma relação {qualifica(corrs['pearson'])} (linear) "
        f"e {qualifica(corrs['spearman'])} (monotônica)."
    )
    return " ".join(partes)

def calcular_diagnosticos(df: pd.DataFrame, coluna_x: str, coluna_y: str) -> dict:
    serie_x = df[coluna_x].dropna()
    serie_y = df[coluna_y].dropna()
    serie_x, serie_y = serie_x.align(serie_y, join="inner")

    resultados = {}

    # Outliers bivariados via regressão linear simples (numpy)
    coef = np.polyfit(serie_x, serie_y, 1)  # coef[0]=inclinação, coef[1]=intercepto
    pred = np.polyval(coef, serie_x)
    residuos = serie_y - pred
    outliers = np.sum(np.abs(residuos) > 3 * residuos.std())
    resultados["outliers_bivariados"] = int(outliers)

    # Heterocedasticidade: variância condicional em quantis de X
    grupos = pd.qcut(serie_x, q=4, duplicates="drop")
    variancias = serie_y.groupby(grupos).var()
    resultados["heterocedasticidade"] = variancias.max() > 2 * variancias.min()

    return resultados

def interpretar_diagnosticos(diag: dict) -> str:
    """Interpreta os diagnósticos adicionais do scatterplot."""
    partes = []
    if diag["outliers_bivariados"] > 0:
        partes.append(f"Foram detectados {diag['outliers_bivariados']} pontos fora do padrão esperado (outliers bivariados).")
    if diag["heterocedasticidade"]:
        partes.append("A dispersão dos pontos varia conforme X (heterocedasticidade).")
    if not partes:
        partes.append("Não foram detectados problemas relevantes de outliers bivariados ou heterocedasticidade.")
    return " ".join(partes)

def interpretar_inclinacao(corr: float, coluna_x: str, coluna_y: str) -> str:
    if corr > 0:
        return f"A relação entre '{coluna_x}' e '{coluna_y}' é positiva: à medida que {coluna_x} aumenta, {coluna_y} tende a aumentar."
    elif corr < 0:
        return f"A relação entre '{coluna_x}' e '{coluna_y}' é negativa: à medida que {coluna_x} aumenta, {coluna_y} tende a diminuir."
    else:
        return f"Não há inclinação clara entre '{coluna_x}' e '{coluna_y}'."

def detectar_clusters(df: pd.DataFrame, coluna_x: str, coluna_y: str, max_clusters: int = 4) -> str:
    X = df[[coluna_x, coluna_y]].dropna().to_numpy()
    if len(X) < 10:
        return "Poucos dados para avaliar clusters."
    
    # Testa de 2 até max_clusters
    melhor_k = None
    melhor_inercia = None
    for k in range(2, max_clusters+1):
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        kmeans.fit(X)
        if melhor_inercia is None or kmeans.inertia_ < melhor_inercia * 0.9:
            melhor_k = k
            melhor_inercia = kmeans.inertia_
    
    if melhor_k and melhor_k > 1:
        return f"Foram identificados aproximadamente {melhor_k} agrupamentos distintos de pontos (clusters)."
    else:
        return "Não foram identificados agrupamentos claros de pontos."

# -----------------------------------
# Função principal para scatterplot
# -----------------------------------

def scatterplot_maker(
    coluna_x: str,
    coluna_y: str,
    title: str = None,
    xlabel: str = None,
    ylabel: str = None
) -> str:
    """Gera instruções para plotar um scatterplot e inclui interpretação estatística da relação."""

    df = st.session_state.get("df_pandas_sample")
    if df is None:
        return json.dumps({"mensagem": "Nenhuma amostra disponível.", "status": "error"})

    for col in [coluna_x, coluna_y]:
        if col not in df.columns:
            return json.dumps({"mensagem": f"A coluna '{col}' não existe na amostra.", "status": "error"})
        if not pd.api.types.is_numeric_dtype(df[col]):
            return json.dumps({"mensagem": f"A coluna '{col}' não é numérica e não pode ser usada em scatterplot.", "status": "error"})

    # Calcular correlações e diagnósticos
    corrs = calcular_correlacoes(df, coluna_x, coluna_y)
    interpretacao_corr = interpretar_correlacoes(corrs, coluna_x, coluna_y)

    interpretacao_inclinacao = interpretar_inclinacao(corrs["pearson"], coluna_x, coluna_y)
    interpretacao_clusters = detectar_clusters(df, coluna_x, coluna_y)

    diag = calcular_diagnosticos(df, coluna_x, coluna_y)
    interpretacao_diag = interpretar_diagnosticos(diag)

    # Definições padrão
    title = title or f"Relação entre {coluna_x} e {coluna_y}"
    xlabel = xlabel or coluna_x
    ylabel = ylabel or coluna_y

    mensagem = (
    f"Scatterplot mostrando a relação entre '{coluna_x}' e '{coluna_y}'.\n\n"
    f"{interpretacao_corr}\n\n"
    f"{interpretacao_inclinacao}\n\n"
    f"{interpretacao_diag}\n\n"
    f"{interpretacao_clusters}\n\n"
    f"📊 O gráfico abaixo ilustra essa relação."
    )

    scatter_inst = {
        "mensagem": mensagem,
        "acao": "scatterplot",
        "args": {
            "coluna_x": coluna_x,
            "coluna_y": coluna_y,
            "title": title,
            "xlabel": xlabel,
            "ylabel": ylabel
        },
        "status": "success"
    }
    return json.dumps(scatter_inst)


# -----------------------------------
# Funções auxiliares para análise de clusters
# -----------------------------------

def interpretar_clusters(medias: pd.DataFrame, df_clusters: pd.DataFrame, min_cluster_size: int) -> str:
    interpretacoes = []

    # 1. Variáveis que mais diferenciam clusters
    variacoes = (medias.max() - medias.min()).sort_values(ascending=False)
    top_vars = variacoes.head(3).index.tolist()
    interpretacoes.append(
        f"As variáveis que mais diferenciam os clusters são: {', '.join(top_vars)}."
    )

    # 2. Perfis médios por cluster
    medias_globais = medias.mean()
    for cluster_id, row in medias.iterrows():
        acima = [col for col in medias.columns if row[col] > medias_globais[col]]
        abaixo = [col for col in medias.columns if row[col] < medias_globais[col]]
        interpretacoes.append(
            f"O Cluster {cluster_id} apresenta valores acima da média em {', '.join(acima) or 'nenhuma variável'} "
            f"e abaixo da média em {', '.join(abaixo) or 'nenhuma variável'}."
        )

    # 3. Clusters semelhantes ou distintos
    distancias = medias.apply(lambda x: np.linalg.norm(x - medias.mean()), axis=1)
    mais_proximos = distancias.sort_values().index.tolist()
    if len(mais_proximos) >= 2:
        interpretacoes.append(
            f"Os clusters {mais_proximos[0]} e {mais_proximos[1]} possuem perfis médios bastante semelhantes."
        )

    # 4. Identificação de outliers coletivos
    counts = df_clusters["cluster"].value_counts()
    clusters_pequenos = counts[counts < max(min_cluster_size, 5)].index.tolist()
    if clusters_pequenos:
        interpretacoes.append(
            f"Os clusters {', '.join(map(str, clusters_pequenos))} possuem poucas observações e podem representar grupos atípicos ou ruído."
        )

    return "\n\n".join(interpretacoes)

# -----------------------------------
# Função principal para análise de clusters
# -----------------------------------

def cluster_analyzer(
    max_clusters: int = 6,
    random_state: int = 42,
    min_cluster_size: int = 0,
    scaling: str = "standard"
) -> str:
    df = st.session_state.get("df_pandas_sample")
    if df is None:
        return json.dumps({"mensagem": "Nenhuma amostra disponível.", "status": "error"})

    # Seleciona apenas colunas numéricas
    num_df = df.select_dtypes(include=[np.number]).dropna()
    if num_df.shape[1] < 2:
        return json.dumps({"mensagem": "Poucas variáveis numéricas para clustering.", "status": "error"})

    # Normalização
    if scaling == "standard":
        scaler = StandardScaler()
        X = scaler.fit_transform(num_df)
    elif scaling == "minmax":
        scaler = MinMaxScaler()
        X = scaler.fit_transform(num_df)
    else:  # "none"
        X = num_df.to_numpy()

    # Busca melhor número de clusters
    melhor_k, melhor_score, melhor_modelo = None, -1, None
    for k in range(2, min(max_clusters, len(X))):
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        if score > melhor_score:
            melhor_k, melhor_score, melhor_modelo = k, score, kmeans

    if melhor_modelo is None:
        return json.dumps({"mensagem": "Não foi possível identificar clusters.", "status": "error"})

    labels = melhor_modelo.labels_
    df_clusters = num_df.copy()
    df_clusters["cluster"] = labels

    # Aplica filtro de tamanho mínimo de cluster
    if min_cluster_size > 0:
        counts = df_clusters["cluster"].value_counts()
        clusters_validos = counts[counts >= min_cluster_size].index
        df_clusters = df_clusters[df_clusters["cluster"].isin(clusters_validos)]

    if df_clusters.empty:
        return json.dumps({"mensagem": "Todos os clusters foram descartados pelo filtro de tamanho mínimo.", "status": "error"})

    # Médias por cluster
    medias = df_clusters.groupby("cluster").mean().round(2)

    # Interpretação enriquecida
    interpretacoes_extra = interpretar_clusters(medias, df_clusters, min_cluster_size)

    mensagem = (
        f"Foram identificados **{medias.shape[0]} clusters** nos dados, "
        f"com um índice de silhueta de {melhor_score:.2f}.\n\n"
        f"O heatmap abaixo mostra as médias das variáveis numéricas em cada cluster.\n\n"
        f"{interpretacoes_extra}"
    )

    cluster_inst = {
        "mensagem": mensagem,
        "acao": "heatmap_clusters",
        "args": {
            "medias": medias.to_dict(),
            "variaveis": list(medias.columns),
            "clusters": list(medias.index)
        },
        "status": "success"
    }
    return json.dumps(cluster_inst)

# -----------------------------------------------
# Funções auxiliares para análise de cross-tab
# -----------------------------------------------
def gerar_crosstab(df, coluna_x, coluna_y, normalize=False):
    """
    Gera a tabela cruzada e retorna também estatísticas úteis.
    """
    ct = pd.crosstab(df[coluna_x], df[coluna_y])
    if normalize:
        ct_norm = ct / ct.values.sum()
    else:
        ct_norm = None

    # Estatísticas básicas
    total = ct.values.sum()
    row_totals = ct.sum(axis=1).to_dict()
    col_totals = ct.sum(axis=0).to_dict()

    return {
        "crosstab": ct,
        "crosstab_norm": ct_norm,
        "total": int(total),
        "row_totals": row_totals,
        "col_totals": col_totals
    }

def calcular_associacao(ct: pd.DataFrame):
    """
    Calcula teste qui-quadrado para verificar associação entre variáveis.
    """
    chi2, p, dof, expected = chi2_contingency(ct)
    return {
        "chi2": chi2,
        "p_value": p,
        "graus_liberdade": dof,
        "significativo": p < 0.05
    }

def interpretar_crosstab(ct: pd.DataFrame, stats: dict, coluna_x: str, coluna_y: str):
    """
    Gera uma interpretação textual da tabela cruzada e do teste estatístico.
    """
    insights = []

    # Categoria mais frequente
    max_cell = ct.stack().idxmax()
    max_value = ct.stack().max()
    insights.append(
        f"A combinação mais frequente é **{coluna_x}={max_cell[0]}** com **{coluna_y}={max_cell[1]}**, "
        f"ocorrendo {max_value} vezes."
    )

    # Categoria dominante em cada linha
    for row in ct.index:
        col_dominante = ct.loc[row].idxmax()
        val = ct.loc[row].max()
        insights.append(
            f"Na categoria **{coluna_x}={row}**, a maior concentração está em **{coluna_y}={col_dominante}** ({val} ocorrências)."
        )

    # Teste de associação
    if stats["significativo"]:
        insights.append(
            f"O teste qui-quadrado indica associação significativa entre **{coluna_x}** e **{coluna_y}** "
            f"(χ²={stats['chi2']:.2f}, p={stats['p_value']:.4f})."
        )
    else:
        insights.append(
            f"O teste qui-quadrado não encontrou associação estatisticamente significativa "
            f"entre **{coluna_x}** e **{coluna_y}** (p={stats['p_value']:.4f})."
        )

    return "\n".join(insights)

# -------------------------------
# Função principal para analise de cross-tab
# -------------------------------

def crosstab_analyzer(
    coluna_x: str,
    coluna_y: str,
    normalize: bool = False,
    output: str = "table",
    max_table_dim: int = 10  # limite para renderizar tabela (linhas e colunas)
) -> str:
    df = st.session_state.get("df_pandas_sample")
    if df is None:
        return json.dumps({"mensagem": "Nenhuma amostra disponível.", "status": "error"})

    if coluna_x not in df.columns or coluna_y not in df.columns:
        return json.dumps({"mensagem": f"Colunas {coluna_x} ou {coluna_y} não encontradas.", "status": "error"})

    # Checagem de cardinalidade para decidir renderização
    n_rows = df[coluna_x].nunique(dropna=False)
    n_cols = df[coluna_y].nunique(dropna=False)

    # Se tabela for muito grande, força heatmap para evitar mandar muitos dados
    render_output = output
    if output == "table" and (n_rows > max_table_dim or n_cols > max_table_dim):
        render_output = "heatmap"

    # Gera crosstab e interpretações
    ct_info = gerar_crosstab(df, coluna_x, coluna_y, normalize)
    stats = calcular_associacao(ct_info["crosstab"])
    interpretacao = interpretar_crosstab(ct_info["crosstab"], stats, coluna_x, coluna_y)

    # Mensagem final já com interpretação embutida
    mensagem = (
        f"Tabela cruzada entre **{coluna_x}** e **{coluna_y}** "
        f"{'(valores normalizados)' if normalize else '(contagens absolutas)'}."
        f"{' (Tabela grande: exibida como heatmap.)' if render_output == 'heatmap' and output == 'table' else ''}\n\n"
        f"**Interpretação automática:**\n{interpretacao}"
    )

    result = {
        "mensagem": mensagem,
        "acao": "crosstab",  # ploter único
        "args": {
            "coluna_x": coluna_x,
            "coluna_y": coluna_y,
            "normalize": normalize,
            "output": render_output,
            "max_table_dim": max_table_dim,
            "shape_hint": {"rows": n_rows, "cols": n_cols}
        },
        "status": "success"
    }
    return json.dumps(result)

# -----------------------------------
# Funções auxiliares para matriz de correlação
# -----------------------------------

def calcular_matriz_correlacao(df: pd.DataFrame, metodo: str = "pearson") -> pd.DataFrame:
    """
    Calcula a matriz de correlação entre variáveis numéricas.
    """
    num_df = df.select_dtypes(include=np.number)
    if num_df.empty:
        return pd.DataFrame()
    return num_df.corr(method=metodo)

def extrair_top_correlacoes(corr_matrix: pd.DataFrame, top_n: int = 5) -> list:
    """
    Extrai os top_n pares de variáveis mais correlacionados (positivos e negativos).
    Retorna lista de tuplas: (col1, col2, valor).
    """
    corr_pairs = (
        corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        .stack()
        .reset_index()
    )
    corr_pairs.columns = ["var1", "var2", "correlacao"]
    corr_pairs["abs_corr"] = corr_pairs["correlacao"].abs()
    top_pairs = corr_pairs.sort_values("abs_corr", ascending=False).head(top_n)
    return list(top_pairs[["var1", "var2", "correlacao"]].itertuples(index=False, name=None))

def interpretar_correlacoes_matrix(corr_matrix: pd.DataFrame, top_n: int = 5) -> str:
    """
    Gera interpretação textual da matriz de correlação.
    """
    if corr_matrix.empty:
        return "Não foi possível calcular correlações, pois não há variáveis numéricas."

    top_pairs = extrair_top_correlacoes(corr_matrix, top_n=top_n)

    mensagens = ["Análise de correlação entre variáveis numéricas:"]
    for var1, var2, corr in top_pairs:
        if corr > 0.7:
            intensidade = "forte positiva"
        elif corr > 0.4:
            intensidade = "moderada positiva"
        elif corr > 0.2:
            intensidade = "fraca positiva"
        elif corr < -0.7:
            intensidade = "forte negativa"
        elif corr < -0.4:
            intensidade = "moderada negativa"
        elif corr < -0.2:
            intensidade = "fraca negativa"
        else:
            intensidade = "muito fraca ou inexistente"

        mensagens.append(
            f"- **{var1}** e **{var2}** apresentam correlação {intensidade} ({corr:.2f})."
        )

    return "\n".join(mensagens)


# -----------------------------------
# Função principal para matriz de correlação
# -----------------------------------
def correlation_matrix(method: str = "pearson", top_n: int = 3) -> str:
    df = st.session_state.get("df_pandas_sample")
    if df is None:
        return json.dumps({"mensagem": "Nenhuma amostra disponível.", "status": "error"})

    num_df = df.select_dtypes(include="number")
    if num_df.shape[1] < 2:
        return json.dumps({"mensagem": "Não há colunas numéricas suficientes para calcular correlação.", "status": "error"})

    corr = num_df.corr(method=method)

    # Identificar pares mais fortes
    corr_pairs = (
        corr.where(~np.tril(np.ones(corr.shape)).astype(bool))
        .stack()
        .sort_values(key=lambda x: x.abs(), ascending=False)
    )
    top_pairs = corr_pairs.head(top_n)


    # Interpretação textual usando função auxiliar
    mensagem = interpretar_correlacoes_matrix(corr, top_n=top_n)


    result = {
        "mensagem": mensagem,
        "acao": "correlation_matrix",
        "args": {
            "method": method  # 👈 só o necessário para o ploter recalcular
        },
        "status": "success"
    }
    return json.dumps(result)

# -----------------------------------
# Função auxiliar para tendências temporais
# -----------------------------------

def calcular_delta_inicial_final(serie: pd.Series) -> tuple:
    """
    Calcula a diferença entre o valor inicial e final de uma série temporal.
    Retorna (valor_inicial, valor_final, delta).
    """
    serie = serie.dropna()
    if serie.shape[0] < 2:
        return None, None, None
    return serie.iloc[0], serie.iloc[-1], serie.iloc[-1] - serie.iloc[0]


def interpretar_tendencia_variavel(serie: pd.Series, nome_coluna: str) -> str:
    """
    Gera interpretação textual para uma variável temporal.
    """
    inicial, final, delta = calcular_delta_inicial_final(serie)
    if inicial is None:
        return f"- A variável **{nome_coluna}** não possui dados suficientes para análise temporal."
    
    if delta > 0:
        direcao = "aumentou"
    elif delta < 0:
        direcao = "diminuiu"
    else:
        direcao = "permaneceu estável"

    return f"- A variável **{nome_coluna}** {direcao} de {inicial:.2f} para {final:.2f}."


def interpretar_tendencias_temporais(ts: pd.DataFrame, freq: str, agg: str) -> str:
    """
    Gera interpretação textual para múltiplas variáveis temporais.
    """
    mensagens = [f"Evolução temporal das variáveis {', '.join(ts.columns)} agregadas por {freq} usando {agg}:"]
    for col in ts.columns:
        mensagens.append(interpretar_tendencia_variavel(ts[col], col))
    return "\n".join(mensagens)

# -----------------------------------
# Funcao principal para tendências temporais
# -----------------------------------

def temporal_trends(
    coluna_tempo: str,
    variaveis: list,
    freq: str = "D",
    agg: str = "mean"
) -> str:
    df = st.session_state.get("df_pandas_sample")
    if df is None:
        return json.dumps({"mensagem": "Nenhuma amostra disponível.", "status": "error"})

    if coluna_tempo not in df.columns:
        return json.dumps({"mensagem": f"Coluna temporal {coluna_tempo} não encontrada.", "status": "error"})

    # Converter para datetime
    df[coluna_tempo] = pd.to_datetime(df[coluna_tempo], errors="coerce")
    if df[coluna_tempo].isna().all():
        return json.dumps({"mensagem": f"A coluna {coluna_tempo} não contém valores de data válidos.", "status": "error"})

    # Selecionar variáveis numéricas válidas
    num_cols = [c for c in variaveis if c in df.select_dtypes(include="number").columns]
    if not num_cols:
        return json.dumps({"mensagem": "Nenhuma variável numérica válida selecionada.", "status": "error"})

    # Agregação temporal
    ts = df.set_index(coluna_tempo).resample(freq)[num_cols].agg(agg)

    # 👉 Interpretação textual usando função auxiliar
    mensagem = interpretar_tendencias_temporais(ts, freq, agg)

    result = {
        "mensagem": mensagem,
        "acao": "temporal_trends",
        "args": {
            "coluna_tempo": coluna_tempo,
            "variaveis": num_cols,
            "freq": freq,
            "agg": agg
        },
        "status": "success"
    }
    return json.dumps(result)

# -----------------------------------
# Funções auxiliares para análise de feature importance
# -----------------------------------
def interpretar_feature_importance(importances: list, target: str, method: str) -> str:
    """
    Gera interpretação textual a partir da lista de importâncias.
    - importances: lista de tuplas (variável, importância)
    - target: variável alvo
    - method: método usado ('linear' ou 'tree')
    """
    if not importances:
        return f"Não foi possível calcular importâncias para prever **{target}**."

    mensagens = [f"Análise de importância de variáveis para prever **{target}** usando método **{method}**:"]

    # Destacar a variável mais importante
    mais_importante, maior_valor = importances[0]
    mensagens.append(f"A variável mais influente foi **{mais_importante}** (importância {maior_valor:.2f}).")

    # Listar todas em ordem
    for feat, imp in importances:
        mensagens.append(f"- **{feat}**: importância relativa {imp:.2f}")

    return "\n".join(mensagens)


# -----------------------------------
# Função principal para análise de feature importance
# -----------------------------------

def feature_importance_analyzer(
    target: str,
    features: list,
    method: str = "linear",
    max_depth: int = 4,
    random_state: int = 42
) -> str:
    df = st.session_state.get("df_pandas_sample")
    if df is None:
        return json.dumps({"mensagem": "Nenhuma amostra disponível.", "status": "error"})

    if target not in df.columns:
        return json.dumps({"mensagem": f"Coluna alvo {target} não encontrada.", "status": "error"})

    valid_features = [f for f in features if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]
    if not valid_features:
        return json.dumps({"mensagem": "Nenhuma variável numérica válida encontrada entre as features.", "status": "error"})

    X = df[valid_features].dropna()
    y = df.loc[X.index, target].dropna()
    X = X.loc[y.index]

    if method == "linear":
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = LinearRegression()
        model.fit(X_scaled, y)
        importances = dict(zip(valid_features, model.coef_))
        importances = {k: abs(v) / sum(abs(val) for val in importances.values()) for k, v in importances.items()}
    else:
        model = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
        model.fit(X, y)
        importances = dict(zip(valid_features, model.feature_importances_))

    # Ordenar importâncias
    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)

    # 👉 Interpretação textual usando função auxiliar
    mensagem = interpretar_feature_importance(sorted_imp, target, method)

    result = {
        "mensagem": mensagem,
        "acao": "feature_importance",
        "args": {
            "importances": sorted_imp,
            "target": target,
            "method": method
        },
        "status": "success"
    }
    return json.dumps(result)

# -----------------------------------
# Ferramentas auxiliares de frequencia_valores
# -----------------------------------

def interpretar_frequencias(freq: pd.Series, coluna: str, top_n: int = 5) -> str:
    """
    Gera interpretação textual para os valores mais e menos frequentes de uma coluna.
    """
    if freq.empty:
        return f"A coluna **{coluna}** não possui valores válidos."

    top_vals = freq.head(top_n)
    bottom_vals = freq.tail(top_n)

    mensagens = [f"Análise de frequência da coluna **{coluna}**:"]

    # Destacar o valor mais frequente
    mais_comum, mais_freq = top_vals.index[0], top_vals.iloc[0]
    mensagens.append(f"O valor mais frequente foi **{mais_comum}** ({mais_freq} ocorrências).")

    # Listar top N
    mensagens.append("Valores mais frequentes:")
    for idx, val in top_vals.items():
        mensagens.append(f"- {idx}: {val} ocorrências")

    # Listar menos frequentes
    mensagens.append("Valores menos frequentes:")
    for idx, val in bottom_vals.items():
        mensagens.append(f"- {idx}: {val} ocorrências")

    return "\n".join(mensagens)

# -----------------------------------
# Função principal para frequencia_valores
# -----------------------------------
def frequencia_valores(coluna: str, top_n: int = 5) -> str:
    """
    Analisa a frequência de valores em uma coluna (categórica ou numérica discreta).
    Retorna os valores mais e menos frequentes.
    """
    df = st.session_state.get("df_pandas_sample")
    if df is None:
        return json.dumps({"mensagem": "Nenhuma amostra disponível.", "status": "error"})

    if coluna not in df.columns:
        return json.dumps({"mensagem": f"Coluna {coluna} não encontrada.", "status": "error"})

    freq = df[coluna].value_counts(dropna=False)

    if freq.empty:
        return json.dumps({"mensagem": f"A coluna {coluna} não possui valores válidos.", "status": "error"})

    # 👉 Interpretação textual usando função auxiliar
    mensagem = interpretar_frequencias(freq, coluna, top_n=top_n)

    result = {
        "mensagem": mensagem,
        "acao": "frequencia_valores",
        "args": {
            "coluna": coluna,
            "top_n": top_n
        },
        "status": "success"
    }
    return json.dumps(result)

# -----------------------------------
# Lista de ferramentas
# -----------------------------------

tools_list = [
    StructuredTool.from_function(
        func=get_columns,
        name="colunas_dataframe",
        description="Lista as colunas disponíveis no DataFrame.",
        args_schema=SemArgs
    ),
    StructuredTool.from_function(
        func=get_summary,
        name="resumo_estatistico",
        description="Mostra estatísticas básicas do DataFrame inteiro ou de uma coluna específica.",
        args_schema=ColunaInput
    ),
    StructuredTool.from_function(
        func=get_data_types,
        name="tipos_dados",
        description="Informa os tipos de cada coluna, separando numéricas e categóricas.",
        args_schema=SemArgs
    ),
    StructuredTool.from_function(
        func=get_distributions,
        name="distribuicoes",
        description="Mostra distribuições de valores: estatísticas para numéricas, contagem para categóricas.",
        args_schema=ColunaInput
    ),
    StructuredTool.from_function(
        func=get_ranges,
        name="intervalos",
        description="Mostra mínimo e máximo das colunas numéricas.",
        args_schema=ColunaInput
    ),
    StructuredTool.from_function(
        func=get_central_tendency,
        name="tendencia_central",
        description="Calcula média, mediana e moda de uma coluna numérica ou de todas as colunas numéricas.",
        args_schema=ColunaInput
    ),
    StructuredTool.from_function(
        func=get_variability,
        name="variabilidade",
        description="Calcula desvio padrão e variância de uma coluna numérica ou de todas as colunas numéricas.",
        args_schema=ColunaInput
    ),
    StructuredTool.from_function(
    func=histogram_maker,
    name="histograma_interativo",
    description="Gera um dicionário de informações que serão usados para plotar um histograma pela interface Streamlit. Aceita parâmetros para personalização do gráfico.",
    args_schema=HistogramInput,
    return_direct=True
    ),
    StructuredTool.from_function(
    func=outlier_analyzer,
    name="outlier_analyzer",
    description=(
        "Analisa uma coluna numérica para identificar outliers usando a regra do IQR. "
        "Retorna um dicionário contendo uma mensagem interpretativa e instruções para "
        "plotar um boxplot na interface Streamlit. "
        "Aceita parâmetros opcionais para personalização do gráfico e controle da análise "
        "(multiplicador do IQR, inclusão de resumo estatístico e recomendações). "
        "Também permite aplicar tratamentos especiais: remover outliers, aplicar transformação logarítmica "
        "ou winsorizar os dados, sempre comparando estatísticas com a versão normal."),
    args_schema=OutlierAnalyzerInput,
    return_direct=True
    ),
    StructuredTool.from_function(
    func=scatterplot_maker,
    name="scatterplot",
    description=(
        "Gera um dicionário de informações que será usado para plotar um gráfico de dispersão "
        "(scatterplot) entre duas colunas numéricas na interface Streamlit. "
        "Além do gráfico, calcula correlações de Pearson e Spearman, detecta outliers bivariados "
        "e avalia heterocedasticidade, retornando uma interpretação textual junto ao gráfico." ),
    args_schema=ScatterplotInput,
    return_direct=True
    ),
    StructuredTool.from_function(
    func=cluster_analyzer,
    name="cluster_analyzer",
    description=(
            "Executa análise de clusterização geral com KMeans em múltiplas variáveis numéricas. "
            "Aceita parâmetros opcionais: max_clusters (padrão 6), random_state (padrão 42), "
            "min_cluster_size (descarta clusters pequenos), scaling ('standard', 'minmax' ou 'none'). "
            "Retorna interpretação textual e instruções para plotar um heatmap de médias por cluster."),
    args_schema=ClusterAnalyzerInput,
    return_direct=True
    ),
    StructuredTool.from_function(
    func=crosstab_analyzer,
    name="crosstab_analyzer",
    args_schema=CrosstabAnalyzerInput,
    description=(
        "Gera uma tabela cruzada entre duas variáveis categóricas. "
        "Pode retornar em formato de tabela ('table') ou como heatmap ('heatmap'). "
        "Se a cardinalidade for muito alta, força automaticamente o modo heatmap. "
        "Aceita parâmetro normalize=True para mostrar proporções em vez de contagens absolutas."),
    return_direct=True
    ),
    StructuredTool.from_function(
    func=correlation_matrix,
    name="correlation_matrix",
    args_schema=CorrelationMatrixInput,
    description=(
        "Calcula a matriz de correlação entre todas as variáveis numéricas. "
        "Retorna interpretação textual destacando os pares mais correlacionados "
        "e instruções para plotar um heatmap da matriz de correlação."),
    return_direct=True
    ),
    StructuredTool.from_function(
        func=temporal_trends,
        name="temporal_trends",
        args_schema=TemporalTrendsInput,
        description=(
            "Analisa a evolução temporal de variáveis numéricas em relação a uma coluna de tempo. "
            "Permite escolher frequência de agregação (diária, mensal, anual) e função de agregação (média, soma, etc.). "
            "Retorna interpretação textual e instruções para plotar séries temporais na interface Streamlit."),
        return_direct=True
    ),
    StructuredTool.from_function(
        func=feature_importance_analyzer,
        name="feature_importance_analyzer",
        args_schema=FeatureImportanceInput,
        description=(
            "Mede a importância relativa de variáveis independentes para prever uma variável alvo. "
            "Pode usar regressão linear múltipla (coeficientes padronizados) ou árvore de decisão (feature importance). "
            "Retorna interpretação textual e instruções para plotar gráfico de barras com as importâncias."),
        return_direct=True
    ),
    StructuredTool.from_function(
        func=frequencia_valores,
        name="frequencia_valores",
        args_schema=FrequenciaValoresInput,
        description=(
            "Analisa a frequência de valores em uma coluna categórica ou numérica discreta. "
            "Retorna os N valores mais frequentes e pode plotar um gráfico de barras."),
        return_direct=True
    ),

]


