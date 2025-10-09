import streamlit as st
import os
import dask.dataframe as dd

# =======================
# Funções utilitárias necessárias
# =======================

def aplicar_acoes_dask_lazy(df_dask, acoes: list):
    """
    Aplica ações declarativas em um DataFrame Dask de forma lazy.
    Suporta: renomear, excluir_colunas, dropna(any/all).
    A exclusão de índices é tratada depois no Pandas.
    """
    for acao in acoes:
        tipo = acao.get("tipo")
        if tipo == "renomear":
            df_dask = df_dask.rename(columns={acao["de"]: acao["para"]})
        elif tipo == "excluir_colunas":
            cols = acao.get("colunas", [])
            df_dask = df_dask.drop(columns=[c for c in cols if c in df_dask.columns])
        elif tipo == "dropna":
            modo = acao.get("modo", "any")
            df_dask = df_dask.dropna(how=modo)
        # excluir_indices fica para o Pandas
    return df_dask

def aplicar_excluir_indices_pandas(df_pd, acoes: list):
    """
    Aplica exclusão de índices em um DataFrame Pandas (após compute do Dask).
    """
    for acao in acoes:
        if acao.get("tipo") == "excluir_indices":
            indices = set(acao.get("indices", []))
            df_pd = df_pd.reset_index(drop=True)
            mask = ~df_pd.index.isin(indices)
            df_pd = df_pd.loc[mask]
    return df_pd

# =======================
# Abas originais
# =======================

def aba_exportacao():
    st.subheader("📤 Exportação de Dados")

    if "df_original" not in st.session_state:
        st.warning("⚠️ Nenhum dado disponível para exportação. Vá para a tela de upload primeiro.")
        return

    if "df_edicao" in st.session_state and isinstance(st.session_state.df_edicao, dd.DataFrame):
        df_dask = st.session_state.df_edicao
        acoes = []
    else:
        df_dask = st.session_state.df_original
        acoes = st.session_state.get("acoes", [])

    formato = st.selectbox("Formato", ["CSV", "Parquet", "XLSX"])
    nome_arquivo = st.text_input("Nome do arquivo (sem extensão)", value="dados_editados")

    if st.button("💾 Exportar"):
        try:
            os.makedirs("exportados", exist_ok=True)
            ext = "xlsx" if formato == "XLSX" else formato.lower()
            caminho = os.path.join("exportados", f"{nome_arquivo}.{ext}")

            df_lazy = aplicar_acoes_dask_lazy(df_dask, acoes)
            df_final = df_lazy.compute()
            df_final = aplicar_excluir_indices_pandas(df_final, acoes)

            if formato == "CSV":
                df_final.to_csv(caminho, index=False)
            elif formato == "Parquet":
                df_final.to_parquet(caminho, index=False)
            elif formato == "XLSX":
                df_final.to_excel(caminho, index=False, engine="openpyxl")

            st.success(f"✅ Arquivo exportado com sucesso: `{caminho}`")
            with open(caminho, "rb") as f:
                st.download_button("📥 Baixar arquivo", data=f.read(), file_name=os.path.basename(caminho))
        except Exception as e:
            st.error(f"❌ Erro ao exportar: {e}")


def aba_amostragem():
    st.subheader("🚀 Amostragem para IA")

    if "df_original" not in st.session_state:
        st.warning("⚠️ Nenhum arquivo carregado. Vá para a tela 'Upload' primeiro.")
        return

    if "df_pandas_sample" in st.session_state:
        if st.session_state.get("df_pandas_updated", False):
            st.success("✅ Amostra atualizada e pronta para análise.")
        else:
            st.warning("⚠️ A amostra pode estar desatualizada. Gere novamente.")
    else:
        st.info("ℹ️ Nenhuma amostra gerada ainda.")

    n_linhas = st.number_input("Número de linhas da amostra", min_value=10, max_value=100000, value=1000, step=100)
    tipo_amostra = st.radio("Tipo de amostra", ["Primeiras linhas", "Aleatória"], horizontal=True)

    if st.button("📥 Aplicar amostra"):
        try:
            # Usa df_edicao se existir, senão aplica ações no df_original
            if "df_edicao" in st.session_state and isinstance(st.session_state.df_edicao, dd.DataFrame):
                df_dask = st.session_state.df_edicao
            else:
                df_dask = st.session_state.df_original
                acoes = st.session_state.get("acoes", [])
                df_dask = aplicar_acoes_dask_lazy(df_dask, acoes)

            if tipo_amostra == "Primeiras linhas":
                df_pandas = df_dask.head(n_linhas)
            else:
                total = df_dask.shape[0].compute()
                frac = min(1.0, n_linhas / total)
                df_pandas = df_dask.sample(frac=frac).compute()
                if len(df_pandas) > n_linhas:
                    df_pandas = df_pandas.sample(n=n_linhas)

            st.session_state.df_pandas_sample = df_pandas
            st.session_state.df_pandas_updated = True

            st.success(f"✅ Amostra de {len(df_pandas)} linhas gerada com sucesso.")
            st.dataframe(df_pandas.head(20))
        except Exception as e:
            st.error(f"❌ Erro ao gerar amostra: {e}")


def render():
    st.header("🧱 Consolidação")
    tab1, tab2 = st.tabs(["📤 Exportação", "🚀 Amostragem para IA"])
    with tab1:
        aba_exportacao()
    with tab2:
        aba_amostragem()
