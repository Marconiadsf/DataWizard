'# app.py'
import streamlit as st
from streamlit_option_menu import option_menu

# Importa as telas
import telas.tela_upload as tela_upload
import telas.tela_edicao as tela_edicao
import telas.tela_consolidar as tela_consolidar
import telas.tela_sobre as tela_sobre
import telas.tela_analise as tela_analise
# Configuração da página
st.set_page_config(page_title="DataWizard", layout="wide")

# Menu lateral
with st.sidebar:
    escolha = option_menu(
        menu_title="Navegação",
        options=["Upload", "Edição", "Consolidar", "Análise IA", "Sobre"],
        icons=["cloud-upload", "pencil", "download","robot", "info-circle"],
        menu_icon="cast",
        default_index=0,
    )
    st.markdown("---")
    st.subheader("⚙️ Configurações de IA")

    # Campo para API Key
    api_key = st.text_input("🔑 API Key", type="password")
    st.session_state["api_key"] = api_key

    # Seletor de modelo
    modelo = st.selectbox("🤖 Modelo de IA", ["Google Gemini", "Groq"])
    st.session_state["modelo_ia"] = modelo
    
    st.markdown("---")
    st.subheader("📋 Ações aplicadas")

    if "acoes" in st.session_state and st.session_state.acoes:
        for i, acao in enumerate(st.session_state.acoes):
            st.markdown(f"**{i+1}.** `{acao['tipo']}`")
            if acao["tipo"] == "renomear":
                st.caption(f"Renomear: {acao['de']} → {acao['para']}")
            elif acao["tipo"] == "excluir_colunas":
                st.caption(f"Excluir colunas: {', '.join(acao['colunas'])}")
            elif acao["tipo"] == "dropna":
                st.caption(f"Excluir nulos: modo `{acao['modo']}`")
            elif acao["tipo"] == "excluir_indices":
                st.caption(f"Excluir linhas por índices: {', '.join(map(str, acao['indices']))}")
    else:
        st.info("Nenhuma ação aplicada ainda.")

# Roteamento entre telas
if escolha == "Upload":
    tela_upload.render()
elif escolha == "Edição":
    tela_edicao.render()
elif escolha == "Consolidar":
    tela_consolidar.render()
elif escolha == "Análise IA":
    tela_analise.render()
elif escolha == "Sobre":
    tela_sobre.render()
