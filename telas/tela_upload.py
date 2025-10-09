import streamlit as st
import dask.dataframe as dd
import os
import csv

def detectar_separador(caminho_arquivo):
    encodings = ["utf-8", "utf-8-sig", "latin1"]
    for enc in encodings:
        try:
            with open(caminho_arquivo, "r", encoding=enc) as f:
                amostra = f.read(2048)
                sniffer = csv.Sniffer()
                dialecto = sniffer.sniff(amostra)
                return dialecto.delimiter, enc
        except Exception:
            continue
    return ",", "utf-8"  # fallback padrão

def render():
    st.header("📤 Upload de Arquivo CSV")

    uploaded_file = st.file_uploader("Selecione um arquivo CSV", type=["csv"])
    if uploaded_file is not None:
        st.warning("⚠️ Processando o arquivo, aguarde...")

        # Salva temporariamente o arquivo
        temp_path = os.path.join("temp", uploaded_file.name)
        os.makedirs("temp", exist_ok=True)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            # Detecta separador e encoding automaticamente
            sep_detectado, encoding_detectado = detectar_separador(temp_path)

            # Permite ao usuário sobrescrever o separador
            st.info(f"📌 Separador detectado: `{sep_detectado}` — Encoding: `{encoding_detectado}`")
            sep_manual = st.selectbox("Deseja sobrescrever o separador?", options=["(usar detectado)", ",", ";", "|", "\t"], index=0)
            sep_final = sep_detectado if sep_manual == "(usar detectado)" else sep_manual

            # Carrega com Dask usando separador e encoding final
            df = dd.read_csv(temp_path, sep=sep_final, encoding=encoding_detectado, assume_missing=True)

            # Armazena no estado
            st.session_state.df_original = df
            st.session_state.df_edicao = df  # compatibilidade visual
            st.session_state.acoes = []      # lista de ações acumuladas
            st.session_state.historico = []  # snapshots para desfazer

            st.success("✅ Dados carregados com sucesso")
            st.dataframe(df.head(100))  # Visualização leve (Dask .head retorna Pandas)
        except Exception as e:
            st.error(f"❌ Erro ao carregar o arquivo: {e}")
