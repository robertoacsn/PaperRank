import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from hybrid_model import extrair_macrotendencias

st.set_page_config(page_title="PaperRank Dashboard", layout="wide")
st.title("PaperRank: Inteligência Híbrida 🚀")
st.markdown("Painel analítico combinando Teoria dos Grafos, BM25, LSI e Critérios de Pesquisa.")

arquivos_csv = glob.glob('paperrank_*.csv')
if not arquivos_csv:
    st.error("Nenhum arquivo de dados encontrado. Rode o 'search.py' primeiro!")
    st.stop()

arquivo_mais_recente = max(arquivos_csv, key=os.path.getctime)
st.success(f"Lendo base de dados mais recente: **{os.path.basename(arquivo_mais_recente)}**")

df = pd.read_csv(arquivo_mais_recente, sep=';')

st.subheader("Top Artigos por Super Score (PageRank + Velocidade)")
colunas_exibicao = ['Title', 'Year', 'Citations', 'Citation_Velocity', 'PageRank', 'Super_Score']
st.dataframe(df[colunas_exibicao].head(10))

# CAMADA DO PESQUISADOR
st.sidebar.header("⚙️ Camada do Pesquisador")
termos_positivos = ['performance', 'accuracy', 'optimization', 'scalable', 'real time', 'solution', 'framework']
termos_negativos = ['review', 'survey', 'challenges', 'barriers', 'limitations', 'future work']

st.sidebar.success(f"**Termos Prioritários (Peso 2.0x):**\n\n{', '.join(termos_positivos)}")
st.sidebar.error(f"**Termos Penalizados (Peso 0.2x):**\n\n{', '.join(termos_negativos)}")

# PROCESSAMENTO HÍBRIDO
df_valid = df.dropna(subset=['Abstract'])
gold_df = df_valid.head(50).copy()

topicos_nomes, topicos_pesos = extrair_macrotendencias(gold_df, termos_positivos, termos_negativos)

# GRÁFICO
st.subheader("Macrotendências Híbridas (BM25 + PageRank + Pesos)")
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(topicos_nomes[::-1], topicos_pesos[::-1], color='darkorange')
ax.set_xlabel('Força Híbrida do Tópico')
ax.set_title('Termos de Impacto Direcionados pelo Pesquisador')
plt.tight_layout()
st.pyplot(fig)