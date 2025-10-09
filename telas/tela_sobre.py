import streamlit as st

def render():
    st.header("ℹ️ Sobre o DataWizard")

    st.markdown("""
    ### 🧠 O que é este app?
    O **DataWizard** é um assistente interativo para explorar, transformar e entender seus dados em CSV.  
    Mais do que apenas abrir arquivos, ele ajuda você a **descobrir padrões, identificar tendências, detectar anomalias e visualizar relações entre variáveis** de forma intuitiva.  

    Com ele, você pode:
    - Carregar seus dados e ter uma visão geral imediata da estrutura e estatísticas principais.
    - Explorar distribuições, frequências e valores atípicos de forma clara e interpretada.
    - Acompanhar a evolução temporal de variáveis e entender como elas mudam ao longo do tempo.
    - Investigar relações entre colunas, correlações e até a influência de variáveis sobre um alvo específico.
    - Criar visualizações automáticas que vêm acompanhadas de interpretações textuais, para que os gráficos não sejam apenas imagens, mas também insights.

    Em resumo, o CSV Wizard funciona como um **parceiro de análise exploratória de dados**, guiando você desde a limpeza e edição até a geração de insights mais profundos.

    ### ⚙️ Tecnologias utilizadas
    O app combina um conjunto robusto de bibliotecas e frameworks:
    - [Streamlit](https://streamlit.io) para a interface web interativa
    - [Dask](https://dask.org) para manipulação distribuída de dados em larga escala
    - [Pandas](https://pandas.pydata.org) para operações tabulares
    - [Scikit-learn](https://scikit-learn.org) e [SciPy](https://scipy.org) para estatística, machine learning e análise de clusters
    - [Matplotlib](https://matplotlib.org) e [Seaborn](https://seaborn.pydata.org) para visualizações
    - [Pydantic](https://docs.pydantic.dev) para validação de dados e schemas
    - [LangChain](https://www.langchain.com) para orquestração de agentes e ferramentas de análise
    - [Google Generative AI (Gemini)](https://ai.google.dev) para geração de linguagem natural e suporte interpretativo
    - [Visual Studio Code](https://code.visualstudio.com) como ambiente de desenvolvimento, integrando e organizando todo o projeto
    - Python 3.11 com suporte a bibliotecas modernas
    - Docker para ambiente isolado e portátil

    ### 📦 Compatibilidade
    - Compatível com arquivos CSV grandes
    - Suporte a múltiplos formatos de CSV (delimitadores, codificações)
    - Suporte a exportação em Parquet e xlsx
    - Funciona em navegadores modernos

    ### 🔗 Links úteis
    - [Repositório no GitHub](https://github.com/Marconiadsf/DataWizard)
    - [Documentação do Dask](https://docs.dask.org/en/latest/)
    - [Guia do Streamlit](https://docs.streamlit.io)

    ---
    """)

    st.subheader("📊 Status atual do app")

    if "df_original" in st.session_state:
        st.success("✅ Arquivo carregado")
        st.write(f"**Colunas:** {list(st.session_state.df_original.columns)}")
        st.write("**Preview:**")
        st.dataframe(st.session_state.df_original.head(5))
    else:
        st.warning("⚠️ Nenhum arquivo carregado ainda.")

    if "df_edicao" in st.session_state:
        st.info("🧹 Edição em andamento")
        st.write(f"**Transformações aplicadas:** {len(st.session_state.historico)}")
    else:
        st.info("📦 Nenhuma edição em andamento")
