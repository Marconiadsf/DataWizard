import streamlit as st
import dask.dataframe as dd
import pandas as pd
import re

def aplicar_acoes_pandas(df_pd: pd.DataFrame, acoes: list) -> pd.DataFrame:
    """
    Aplica ações no DataFrame pandas (usado para preview).
    Suporta: renomear, excluir_colunas, dropna(any/all), excluir_indices (posicional).
    """
    for acao in acoes:
        tipo = acao.get("tipo")
        if tipo == "renomear":
            df_pd = df_pd.rename(columns={acao["de"]: acao["para"]})
        elif tipo == "excluir_colunas":
            cols = acao.get("colunas", [])
            df_pd = df_pd.drop(columns=[c for c in cols if c in df_pd.columns])
        elif tipo == "dropna":
            modo = acao.get("modo", "any")
            df_pd = df_pd.dropna(how=modo)
        elif tipo == "excluir_indices":
            # Índices são posicionalmente relativos ao preview
            indices = set(acao.get("indices", []))
            df_pd = df_pd.reset_index(drop=True)
            mask = ~df_pd.index.isin(indices)
            df_pd = df_pd.loc[mask]
    return df_pd

def render():
    st.header("🧹 Edição")

    if "df_original" not in st.session_state:
        st.warning("⚠️ Nenhum arquivo carregado. Vá para a tela 'Upload' primeiro.")
        return

    # Inicialização de estado declarativo
    if "acoes" not in st.session_state:
        st.session_state.acoes = []
    if "historico" not in st.session_state:
        st.session_state.historico = []
    if "df_edicao" not in st.session_state:
        st.session_state.df_edicao = st.session_state.df_original

    df_dask = st.session_state.df_original  # fonte única de verdade

    # Proteção contra sobrescrita acidental
    if not isinstance(df_dask, dd.DataFrame):
        st.error("❌ O DataFrame foi corrompido. Reiniciando para o original.")
        st.session_state.df_original = st.session_state.df_edicao if isinstance(st.session_state.df_edicao, dd.DataFrame) else None
        st.session_state.df_edicao = st.session_state.df_original
        st.session_state.acoes = []
        st.session_state.historico = []
        st.rerun()

    # Gera preview: Dask .head -> Pandas, aplica ações declarativas e mostra
    try:
        preview_pd = df_dask.head(100)  # retorna Pandas
        preview_pd = aplicar_acoes_pandas(preview_pd, st.session_state.acoes)
    except Exception as e:
        st.error(f"❌ Erro ao gerar preview: {e}")
        return

    st.subheader("📄 Visualização atual")
    st.dataframe(preview_pd)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("🔄 Atualizar visualização"):
            st.rerun()
    with col2:
        if st.button("↩️ Desfazer última ação") and st.session_state.historico:
            # Restaura o snapshot anterior da lista de ações
            st.session_state.acoes = st.session_state.historico.pop()
            st.success("✅ Última ação desfeita.")
            st.rerun()
    with col3:
        if st.button("🗑️ Descartar todas alterações"):
            st.session_state.acoes = []
            st.session_state.historico = []
            st.success("✅ Todas as ações foram descartadas.")
            st.rerun()
    with col4:
        if st.button("✅ Aplicar alterações"):
            try:
                from .tela_consolidar import aplicar_acoes_dask_lazy, aplicar_excluir_indices_pandas
                df_dask = st.session_state.df_original
                acoes = st.session_state.acoes

                # aplica transformações lazy no Dask
                df_lazy = aplicar_acoes_dask_lazy(df_dask, acoes)

                # compute para Pandas
                df_final = df_lazy.compute()

                # aplica exclusão de índices no Pandas
                df_final = aplicar_excluir_indices_pandas(df_final, acoes)

                # guarda como df_edicao (em Dask novamente para consistência)
                st.session_state.df_edicao = dd.from_pandas(df_final, npartitions=1)
                st.success("✅ Alterações aplicadas ao DataFrame completo.")
            except Exception as e:
                st.error(f"❌ Erro ao aplicar alterações: {e}")

    aba = st.tabs(["✏️ Renomear Colunas", "🧹 Limpeza e Exclusões", "💻 Edição Direta"])

    with aba[0]:
        st.markdown("### ✏️ Renomear colunas")
        # As colunas para o seletor vêm do preview atual (após ações),
        # para que o usuário veja os nomes atualizados
        cols_preview = list(preview_pd.columns)
        col_selecionada = st.selectbox("Escolha a coluna para renomear", cols_preview)
        novo_nome = st.text_input("Novo nome para a coluna selecionada", value=col_selecionada)
        if st.button("✅ Renomear coluna"):
            if novo_nome in cols_preview:
                st.error("⚠️ Já existe uma coluna com esse nome.")
            else:
                try:
                    # Snapshot do histórico (lista de ações)
                    st.session_state.historico.append(st.session_state.acoes.copy())
                    # Registra ação declarativa
                    st.session_state.acoes.append({"tipo": "renomear", "de": col_selecionada, "para": novo_nome})
                    st.success(f"✅ Coluna '{col_selecionada}' renomeada para '{novo_nome}' (preview).")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Erro ao renomear: {e}")

    with aba[1]:
        st.markdown("### 🗑️ Exclusão de colunas")
        cols_preview = list(preview_pd.columns)
        cols_to_drop = st.multiselect("Colunas para excluir", cols_preview)
        if st.button("Excluir colunas selecionadas"):
            try:
                st.session_state.historico.append(st.session_state.acoes.copy())
                st.session_state.acoes.append({"tipo": "excluir_colunas", "colunas": cols_to_drop})
                st.success("✅ Colunas marcadas foram excluídas (preview).")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Erro ao excluir colunas: {e}")

        st.markdown("### 🧹 Exclusão de linhas por índice")
        raw_input = st.text_input("Índices para excluir (ex: 1,2,5-10,34-2435)", value="")
        if st.button("Excluir linhas"):
            try:
                indices = parse_indices(raw_input)
                st.session_state.historico.append(st.session_state.acoes.copy())
                st.session_state.acoes.append({"tipo": "excluir_indices", "indices": indices})
                st.success(f"✅ Linhas {indices} marcadas para exclusão (preview).")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Erro ao excluir linhas: {e}")

        st.markdown("### 🚫 Exclusão de nulos")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Excluir linhas totalmente nulas"):
                st.session_state.historico.append(st.session_state.acoes.copy())
                st.session_state.acoes.append({"tipo": "dropna", "modo": "all"})
                st.success("✅ Linhas totalmente nulas excluídas (preview).")
                st.rerun()
        with col2:
            if st.button("Excluir linhas com qualquer nulo"):
                st.session_state.historico.append(st.session_state.acoes.copy())
                st.session_state.acoes.append({"tipo": "dropna", "modo": "any"})
                st.success("✅ Linhas com nulos excluídas (preview).")
                st.rerun()


def parse_indices(text):
    parts = re.split(r"[,\s]+", text.strip())
    indices = []
    for part in parts:
        if "-" in part:
            try:
                start, end = map(int, part.split("-"))
                if start <= end:
                    indices.extend(range(start, end + 1))
            except ValueError:
                continue
        else:
            try:
                if part != "":
                    indices.append(int(part))
            except ValueError:
                continue
    return indices
