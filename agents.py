# =======================
# Bibliotecas da standard library (Python)
# =======================

# Sistema operacional: criação de diretórios, configuração e leitura de variáveis de ambiente
import os
# Expressões regulares
import re

# Manipulação de datas e horários
import datetime

# Execução de funções assíncronas
import asyncio

# Identificadores únicos universais (UUID)
from uuid import uuid4

# Manipulação de CSV
import pandas as pd

# Tipagem estática (anotações e tipos auxiliares)
from typing import Any, List, Union, TypedDict


# Exibição de gráficos e imagens
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# =======================
# Gemini
# =======================

import google.generativeai as gemini

# =======================
# LangChain
# =======================

# Configuração de debug do LangChain
from langchain.globals import set_debug

# Modelos LLM (Large Language Models)
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel

# Construção de prompts
from langchain.schema import HumanMessage
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

# Parsers de saída
from langchain.schema.output_parser import StrOutputParser

# Execução de fluxos (Runnables)
from langchain_core.runnables import RunnableLambda

# Criação e execução de agentes
from langchain.agents import (
    Tool,
    AgentExecutor,
    create_tool_calling_agent,
    create_react_agent
)

# Ferramentas customizadas para agentes
from langchain.tools import tool
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_experimental.tools.python.tool import PythonAstREPLTool

# Tipos de memória utilizados em agentes
from langchain.memory import ConversationBufferMemory

# Acesso ao hub LangChain de prompts prontos (https://smith.langchain.com/hub)
from langchain import hub

# =======================
# MCP (Model Context Protocol)
# =======================

# Servidor MCP
from mcp.server.fastmcp import FastMCP

# Cliente MCP multi-servidor
from langchain_mcp_adapters.client import MultiServerMCPClient

# =======================
# LangGraph
# =======================

# Criação de agentes com LangGraph
from langgraph.prebuilt import create_react_agent as create_react_agent_graph

# Sistema de checkpoint em memória
from langgraph.checkpoint.memory import InMemorySaver

# Definição e execução de grafos
from langgraph.graph import StateGraph, END

# =======================
# Outros
# =======================

import warnings
warnings.filterwarnings('ignore')

# =======================
# Ferramentas customizadas
# =======================
import tools

# =======================
# Streamlit
# =======================
import streamlit as st



#llm_padrao = ChatGroq(temperature=0, groq_api_key=GROQ_API_KEY, model_name='llama-3.3-70b-versatile')                                               
#llm_gemini = ChatGoogleGenerativeAI(temperature=0, api_key=GOOGLE_API_KEY, model='gemini-2.5-flash')                                       
#llm_groq_p = ChatGroq(temperature=0, groq_api_key=GROQ_API_KEY, model_name='deepseek-r1-distill-llama-70b')


# Visualizar os detalhes da execução
set_debug(False)


# =======================
# Prompt do agente
# =======================

agent_prompt = """
Você é um assistente de análise de dados.
Suas ferramentas têm acesso a um DataFrame Pandas chamado `df_pandas_sample`, que representa uma amostra dos dados carregados pelo usuário.

Você possui as seguintes ferramentas:
- colunas_dataframe → lista as colunas disponíveis.
- resumo_estatistico → mostra estatísticas básicas do DataFrame inteiro ou de uma coluna específica.
- tipos_dados → informa os tipos de cada coluna.
- distribuicoes → mostra distribuições de valores (estatísticas para numéricas, contagem para categóricas).
- intervalos → mostra mínimo e máximo das colunas numéricas.
- tendencia_central → retorna média, mediana e moda das colunas numéricas.
- variabilidade → retorna desvio padrão e variância das colunas numéricas.
- histograma_interativo → gera um dicionário de informações que será usado para plotar um histograma pela interface Streamlit. Aceita parâmetros para personalização do gráfico.
- outlier_analyzer → analisa uma coluna numérica para identificar outliers usando a regra do IQR. Retorna um dicionário contendo uma mensagem interpretativa e instruções para plotar um boxplot na interface Streamlit. Aceita parâmetros opcionais para personalização do gráfico e controle da análise (multiplicador do IQR, inclusão de resumo estatístico e recomendações).
- scatterplot → gera um dicionário de informações que será usado para plotar um gráfico de dispersão (scatterplot) entre duas colunas numéricas na interface Streamlit. Além do gráfico, calcula correlações de Pearson e Spearman, detecta outliers bivariados e avalia heterocedasticidade, retornando uma interpretação textual junto ao gráfico.
- cluster_analyzer → executa análise de clusterização geral com KMeans em múltiplas variáveis numéricas. Retorna um dicionário contendo uma mensagem interpretativa e instruções para plotar um heatmap de médias por cluster na interface Streamlit.
- crosstab_analyzer → gera uma tabela cruzada entre duas variáveis categóricas. Retorna um dicionário contendo uma mensagem interpretativa (incluindo insights estatísticos e teste qui-quadrado) e instruções para plotar a saída como tabela ou heatmap na interface Streamlit, dependendo da cardinalidade e do parâmetro de saída.
- correlation_matrix → calcula a matriz de correlação entre todas as variáveis numéricas. Retorna um dicionário contendo uma mensagem interpretativa destacando os pares mais correlacionados e instruções para plotar um heatmap da matriz de correlação na interface Streamlit.
- temporal_trends → analisa a evolução temporal de variáveis numéricas em relação a uma coluna de tempo. Permite escolher frequência de agregação (diária, mensal, anual) e função de agregação (média, soma, etc.). Retorna interpretação textual e instruções para plotar séries temporais na interface Streamlit.
- feature_importance_analyzer → mede a importância relativa de variáveis independentes para prever uma variável alvo. Pode usar regressão linear múltipla (coeficientes padronizados) ou árvore de decisão (feature importance). Retorna interpretação textual e instruções para plotar gráfico de barras com as importâncias.
- frequencia_valores → analisa a frequência de valores em uma coluna categórica ou numérica discreta. Retorna os valores mais e menos frequentes e pode plotar gráfico de barras.

Regras de comportamento:
- Sempre que a pergunta do usuário envolver dados tabulares, utilize as ferramentas adequadas para obter a resposta.
- Sempre que a pergunta envolver visualização gráfica (como histogramas, boxplots, scatterplots, heatmaps de clusters, tabelas/heatmaps de crosstab, matriz de correlação, tendências temporais, importância de variáveis ou frequências), você deve obrigatoriamente invocar a ferramenta correspondente e retornar **exclusivamente** o dicionário produzido por ela, sem adicionar explicações em texto.
- Para perguntas que não envolvem gráficos, explique os resultados em português claro, de forma descritiva, sem citar o nome da ferramenta usada.
- Se a pergunta não exigir cálculo ou consulta, responda diretamente em texto.
- Se o usuário perguntar sobre as ferramentas disponíveis, liste-as em texto simples (sem chamar nenhuma ferramenta).
- Se a linguagem for coloquial ou a pergunta não estiver clara, peça esclarecimentos antes de responder.
- Use apenas as ferramentas listadas acima. Não tente invocar ferramentas inexistentes.
"""


def agente_langchain(llm:BaseChatModel, usar_ferramentas:bool=True) -> dict:
    ferramentas = tools.tools_list if usar_ferramentas else []

    # Retorna o histórico como uma lista de objetos de mensagem
    memoria = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    prompt = ChatPromptTemplate.from_messages(
        [   
            SystemMessagePromptTemplate.from_template(agent_prompt),
            MessagesPlaceholder(variable_name="chat_history"), # O placeholder para o histórico
            HumanMessagePromptTemplate.from_template("{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"), # Onde o agente irá escrever suas anotações (Pensamento)
        ]
    )
    #print([t.name for t in ferramentas])
    agente = create_tool_calling_agent(llm, ferramentas, prompt)
    executor_do_agente = AgentExecutor(agent=agente, tools=ferramentas, memory=memoria)
    return executor_do_agente
# =======================



def get_executor():
    modelo = st.session_state.get("modelo_ia", "Google Gemini")
    api_key = st.session_state.get("api_key", "")

    # Se já existe executor mas modelo ou chave mudaram, reseta
    if "executor_do_agente" in st.session_state:
        if (st.session_state.get("modelo_atual") != modelo or
            st.session_state.get("api_key_atual") != api_key):
            del st.session_state.executor_do_agente

    if "executor_do_agente" not in st.session_state:
        if modelo == "Google Gemini":
            llm = ChatGoogleGenerativeAI(temperature=0, api_key=api_key, model='gemini-2.5-flash')
        elif modelo == "Groq":
            llm = ChatGroq(temperature=0, groq_api_key=api_key, model_name='llama-3.3-70b-versatile')
        else:
            raise ValueError("Modelo de IA desconhecido.")

        st.session_state.executor_do_agente = agente_langchain(llm, usar_ferramentas=True)
        st.session_state["modelo_atual"] = modelo
        st.session_state["api_key_atual"] = api_key

    return st.session_state.executor_do_agente


import json

def makeQuestion(question: str) -> dict:
    executor = get_executor()
    try:
        resposta = executor.invoke({"input": question})
    except Exception as e:
        return {"mensagem": f"Erro ao processar: {e}", "status": "error"}

    # O executor SEMPRE retorna um dict
    output = resposta.get("output")
    
    # Caso 1: ferramenta retornou um dict estruturado
    if isinstance(output, dict):
        return output

    # Caso 2: ferramenta retornou string (texto ou JSON serializado)
    if isinstance(output, str):
        # Tenta interpretar como JSON
        try:
            parsed = json.loads(output)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        # Se não for JSON válido, trata como texto simples
        return {"mensagem": output, "status": "success"}

    # Caso 3: algo inesperado (lista, None, etc.)
    return {"mensagem": str(output), "status": "success"}