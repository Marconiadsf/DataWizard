# tela_analise.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
import agents


import pprint

def debug_resposta(resposta, origem="makeQuestion"):
    print("\n" + "="*60)
    print(f"[DEBUG] Resposta recebida em {origem}:")
    pprint.pprint(resposta, indent=2, width=120)
    print("="*60 + "\n")

def _ensure_chat_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def histogram_ploter(
    coluna: str,
    min_bins: int,
    max_bins: int,
    default_bins: int,
    steps: int,
    edge_color: str,
    title: str,
    xlabel: str,
    ylabel: str,
    idx: int = None,   # novo parâmetro opcional para diferenciar
    *args,
    **kwargs
):
    df = st.session_state.get("df_pandas_sample")
    if df is None:
        st.warning("Nenhuma amostra disponível.")
        return

    if coluna not in df.columns:
        st.warning(f"A coluna '{coluna}' não existe.")
        return

    if not pd.api.types.is_numeric_dtype(df[coluna]):
        st.warning(f"A coluna '{coluna}' não é numérica.")
        return

    data = df[coluna].dropna().values

    # Gera uma chave única para o slider
    key = f"slider_hist_{coluna}"
    if idx is not None:
        key = f"{key}_{idx}"

    # Slider interativo para o usuário escolher o número de bins
    num_bins = st.slider(
        "Selecione o número de intervalos (bins)",
        min_value=min_bins,
        max_value=max_bins,
        value=default_bins,
        step=steps,
        key=key
    )

    # Plot do histograma
    fig, ax = plt.subplots()
    ax.hist(data, bins=num_bins, edgecolor=edge_color)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    st.pyplot(fig)

def boxplot_ploter(
    coluna: str,
    title: str,
    ylabel: str,
    idx: int = None,
    *args,
    **kwargs
):
    df = st.session_state.get("df_pandas_sample")
    if df is None:
        st.warning("Nenhuma amostra disponível.")
        return

    if coluna not in df.columns:
        st.warning(f"A coluna '{coluna}' não existe.")
        return

    if not pd.api.types.is_numeric_dtype(df[coluna]):
        st.warning(f"A coluna '{coluna}' não é numérica.")
        return

    data = df[coluna].dropna().values

    # Gera chave única para o componente (se precisar de interatividade futura)
    key = f"boxplot_{coluna}"
    if idx is not None:
        key = f"{key}_{idx}"

    # Plot do boxplot
    fig, ax = plt.subplots()
    ax.boxplot(data, vert=True, patch_artist=True)
    ax.set_title(title)
    ax.set_ylabel(ylabel)

    st.pyplot(fig)

def scatterplot_ploter(
    coluna_x: str,
    coluna_y: str,
    title: str,
    xlabel: str,
    ylabel: str,
    idx: int = None,
    *args,
    **kwargs
):
    df = st.session_state.get("df_pandas_sample")
    if df is None:
        st.warning("Nenhuma amostra disponível.")
        return

    if coluna_x not in df.columns or coluna_y not in df.columns:
        st.warning("Colunas inválidas para scatterplot.")
        return

    if not pd.api.types.is_numeric_dtype(df[coluna_x]) or not pd.api.types.is_numeric_dtype(df[coluna_y]):
        st.warning("As colunas devem ser numéricas para scatterplot.")
        return

    data_x = df[coluna_x].dropna()
    data_y = df[coluna_y].dropna()

    fig, ax = plt.subplots()
    ax.scatter(data_x, data_y, alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # 🔑 ESSENCIAL: mostrar o gráfico no Streamlit
    st.pyplot(fig)

def heatmap_clusters_ploter(medias: dict, variaveis: list, clusters: list, idx: int = None):
    df_medias = pd.DataFrame(medias, index=clusters)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(df_medias, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Médias por Cluster")
    st.pyplot(fig)
    return ""

def crosstab_ploter(
    coluna_x: str,
    coluna_y: str,
    normalize: bool = False,
    output: str = "table",
    max_table_dim: int = 10,
    shape_hint: dict = None,
    idx: int = None
):
    """
    Plota a tabela cruzada entre duas variáveis categóricas.
    Usa o idx como key para evitar conflitos de renderização no Streamlit.
    """

    df = st.session_state.get("df_pandas_sample")
    if df is None:
        st.warning("Nenhuma amostra disponível para plotar a crosstab.")
        return

    # Gera a crosstab
    ct = pd.crosstab(df[coluna_x], df[coluna_y], normalize="all" if normalize else False)

    if output == "table":
        if ct.shape[0] <= max_table_dim and ct.shape[1] <= max_table_dim:
            st.table(ct)
        else:
            st.dataframe(ct)
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            ct,
            annot=(ct.shape[0] <= 20 and ct.shape[1] <= 20),
            fmt=".2f" if normalize else "d",
            cmap="Blues",
            cbar=True,
            ax=ax
        )
        ax.set_title(f"Heatmap da Tabela Cruzada: {coluna_x} x {coluna_y}", fontsize=14)
        st.pyplot(fig)

def correlation_matrix_ploter(method: str = "pearson", idx: int = None):
    df = st.session_state.get("df_pandas_sample")
    if df is None:
        st.warning("Nenhuma amostra disponível para plotar a matriz de correlação.")
        return

    num_df = df.select_dtypes(include="number")
    corr = num_df.corr(method=method)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        cbar=True,
        ax=ax
    )
    ax.set_title(f"Matriz de Correlação ({method})", fontsize=14)
    st.pyplot(fig)


def temporal_trends_ploter(coluna_tempo: str, variaveis: list, freq: str = "D", agg: str = "mean", idx: int = None):
    df = st.session_state.get("df_pandas_sample")
    if df is None:
        st.warning("Nenhuma amostra disponível para plotar tendências temporais.")
        return

    df[coluna_tempo] = pd.to_datetime(df[coluna_tempo], errors="coerce")
    ts = df.set_index(coluna_tempo).resample(freq)[variaveis].agg(agg)

    fig, ax = plt.subplots(figsize=(10, 6))
    ts.plot(ax=ax)
    ax.set_title(f"Evolução temporal ({agg} por {freq})", fontsize=14)
    ax.set_ylabel("Valor")
    ax.set_xlabel("Tempo")
    st.pyplot(fig)

def feature_importance_ploter(importances: list, target: str, method: str = "linear", idx: int = None):
    feats, vals = zip(*importances)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=list(vals), y=list(feats), ax=ax, palette="viridis")
    ax.set_title(f"Importância das variáveis para prever {target} ({method})", fontsize=14)
    ax.set_xlabel("Importância relativa")
    st.pyplot(fig)

def frequencia_valores_ploter(coluna: str, top_n: int = 5, idx: int = None):
    df = st.session_state.get("df_pandas_sample")
    if df is None:
        st.warning("Nenhuma amostra disponível para plotar frequências.")
        return

    freq = df[coluna].value_counts(dropna=False).head(top_n)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=freq.values, y=freq.index, ax=ax, palette="Blues_r")
    ax.set_title(f"Top {top_n} valores mais frequentes em {coluna}", fontsize=14)
    ax.set_xlabel("Ocorrências")
    st.pyplot(fig)


def render():
    st.title("Análise IA")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Botões de ação
    col1 = st.columns(1)[0]
    with col1:
        if st.button("🗑️ Limpar histórico"):
            st.session_state.chat_history = []
            st.rerun()
   
    # Renderiza histórico
    for i, (autor, conteudo) in enumerate(st.session_state.chat_history):
        if autor == "Você":
            with st.chat_message("user"):
                st.markdown(conteudo)
        else:  # IA
            with st.chat_message("assistant"):
                if isinstance(conteudo, dict):
                    st.markdown(conteudo.get("mensagem", ""))

                    if conteudo.get("acao") == "histograma":
                        histogram_ploter(**conteudo["args"], idx=i)
                    elif conteudo.get("acao") == "boxplot":
                        boxplot_ploter(**conteudo["args"], idx=i)
                    elif conteudo.get("acao") == "scatterplot":
                        scatterplot_ploter(**conteudo["args"], idx=i)
                    elif conteudo.get("acao") == "heatmap_clusters":
                        heatmap_clusters_ploter(**conteudo["args"], idx=i)
                    elif conteudo.get("acao") == "crosstab": 
                        crosstab_ploter(**conteudo["args"],idx=i) 
                    elif conteudo.get("acao") == "correlation_matrix": 
                        correlation_matrix_ploter(**conteudo["args"],idx=i)
                    elif conteudo.get("acao") == "temporal_trends": 
                        temporal_trends_ploter(**conteudo["args"],idx=i)
                    elif conteudo.get("acao") == "feature_importance": 
                        feature_importance_ploter(**conteudo["args"],idx=i)
                    elif conteudo.get("acao") == "frequencia_valores":
                        frequencia_valores_ploter(**conteudo["args"], idx=i)
                    # Mensagem de erro, se houver    
                    if conteudo.get("status") == "error":
                        st.error(conteudo.get("mensagem", "Erro desconhecido."))
                else:
                    st.markdown(conteudo)

    # Entrada do usuário
    if prompt := st.chat_input("Digite sua pergunta sobre o DataFrame:"):
        st.session_state.chat_history.append(("Você", prompt))
        resposta = agents.makeQuestion(prompt)
        st.session_state.chat_history.append(("IA", resposta))
        st.rerun()
