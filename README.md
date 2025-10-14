# 🧙‍♂️ DataWizard – Agente Autônomo para Análise de Dados CSV

**Aplicação online:** [https://iadatawizard.streamlit.app](https://iadatawizard.streamlit.app)

O DataWizard é um agente inteligente desenvolvido com LangChain e Streamlit, capaz de analisar arquivos CSV genéricos, gerar gráficos interativos e responder perguntas com base nos dados. Ele utiliza modelos de linguagem como Google Gemini, Groq e OpenAI, com suporte a ferramentas analíticas e memória conversacional.

---

## 🚀 Guia rápido de uso

1. **Acesse o app online**  
   👉 [https://iadatawizard.streamlit.app](https://iadatawizard.streamlit.app)

2. **Configure o modelo de IA**  
   - Escolha entre Gemini, Groq ou OpenAI na barra lateral.  
   - Insira sua API Key correspondente.

3. **Faça o upload do seu arquivo CSV**  
   - Vá para a aba **Upload**.  
   - Selecione o arquivo e aguarde o carregamento.

4. **Edite os dados (opcional)**  
   - Use a aba **Edição** para aplicar filtros, renomear colunas ou excluir registros.

5. **Consolide os dados (obrigatório para análise com IA)**  
   - A aba **Consolidar** é necessária para preparar os dados antes da análise com IA.  
   - Você pode:
     - Exportar os dados consolidados nos formatos **CSV**, **Parquet** ou **XLSX**.  
     - Ou gerar uma **amostra de dados** para análise com IA na aba seguinte.

6. **Interaja com a IA**  
   - Na aba **Análise IA**, digite perguntas como:  
     - “Qual é a média da coluna Amount?”  
     - “Mostre um histograma da coluna V17.”  
     - “Quais variáveis mais influenciam a coluna Class?”  
     - “Quais conclusões podemos tirar dos dados?”

7. **Sobre o projeto**  
   - A aba **Sobre** traz informações técnicas e créditos.

---

## ⚠️ Aviso importante

> Esta aplicação **não foi testada amplamente em nuvem**.  
> Recomenda-se rodar localmente para maior estabilidade.

---

## 🛠️ Requisitos para rodar localmente

- Python 3.10+
- Instale as dependências:
  ```bash
  pip install -r requirements.txt
- Execute o app:
  ```bash
  streamlit run app.py

##📄 Licença
Este projeto é acadêmico e foi desenvolvido como parte da atividade “Agentes Autônomos – Desafio Extra”.
