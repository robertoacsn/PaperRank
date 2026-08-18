import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.decomposition import TruncatedSVD


# config pagina web
st.set_page_config(page_title="PaperRank Dashboard", layout="wide")
st.title("PaperRank: Inteligência Híbrida 🚀")
st.markdown("Painel analítico combinando Teoria dos Grafos, BM25, LSI e Critérios de Pesquisa.")

# leitura dados
arquivos_csv = glob.glob('paperrank_*.csv')
if not arquivos_csv:
    st.error("Nenhum arquivo de dados encontrado. Rode o 'search.py' primeiro!")
    st.stop()

# pega arquivo mais recente
arquivo_mais_recente = max(arquivos_csv, key=os.path.getctime)
st.success(f"Lendo base de dados mais recente: **{os.path.basename(arquivo_mais_recente)}**")

df = pd.read_csv(arquivo_mais_recente, sep=';')


# interface
st.subheader("Top Artigos por Super Score (PageRank + Velocidade)")
colunas_exibicao = ['Title', 'Year', 'Citations', 'Citation_Velocity', 'PageRank', 'Super_Score']
st.dataframe(df[colunas_exibicao].head(10))

# camada pesquisador
st.sidebar.header("⚙️ Camada do Pesquisador")
st.sidebar.markdown("Critérios humanos que multiplicam ou penalizam o ranqueamento semântico (BM25):")

termos_positivos = ['performance', 'accuracy', 'optimization', 'scalable', 'real time', 'solution', 'framework']
termos_negativos = ['review', 'survey', 'challenges', 'barriers', 'limitations', 'future work']

st.sidebar.success(f"**Termos Prioritários (Peso 2.0x):**\n\n{', '.join(termos_positivos)}")
st.sidebar.error(f"**Termos Penalizados (Peso 0.2x):**\n\n{', '.join(termos_negativos)}")

# motor de busca
df_valid = df.dropna(subset=['Abstract'])
gold_df = df_valid.head(50).copy()

lixo_academico = ['paper', 'presents', 'study', 'results', 'proposed', 'method', 'research', 'based', 'approach', 'article', 'this', 'we', 'show', 'data']
stop_words_finais = list(ENGLISH_STOP_WORDS.union(lixo_academico))

# Frequência Bruta
vectorizer = CountVectorizer(ngram_range=(2, 3), stop_words=stop_words_finais, max_df=0.60, min_df=2)
tf_matrix = vectorizer.fit_transform(gold_df['Abstract']).toarray()
feature_names = vectorizer.get_feature_names_out()

# Matemática BM25
k1 = 1.5 
b = 0.75 
doc_lengths = tf_matrix.sum(axis=1)
avgdl = doc_lengths.mean()
N = len(gold_df)
doc_freqs = (tf_matrix > 0).sum(axis=0)
idf = np.log(((N - doc_freqs + 0.5) / (doc_freqs + 0.5)) + 1)
length_penalty = (1 - b + b * (doc_lengths / avgdl)).reshape(-1, 1)
bm25_matrix = idf * (tf_matrix * (k1 + 1)) / (tf_matrix + k1 * length_penalty)

# Aplicação dos Pesos Híbridos
pesos_pesquisador = np.ones(len(feature_names))
for i, term in enumerate(feature_names):
    if any(pos in term for pos in termos_positivos):
        pesos_pesquisador[i] = 2.0
    elif any(neg in term for neg in termos_negativos):
        pesos_pesquisador[i] = 0.2

bm25_hibrido = bm25_matrix * pesos_pesquisador
super_scores = gold_df['Super_Score'].values
weighted_bm25 = bm25_hibrido * super_scores[:, np.newaxis]

# Modelagem LSI (Tópicos)
NUM_TOPICOS = 8 
lsi_model = TruncatedSVD(n_components=NUM_TOPICOS, random_state=42)
lsi_matrix = lsi_model.fit_transform(weighted_bm25)

topicos_nomes = []
topicos_pesos = lsi_model.singular_values_

for i, comp in enumerate(lsi_model.components_):
    top_words_idx = comp.argsort()[::-1][:3]
    top_words = [feature_names[idx] for idx in top_words_idx]
    nome_topico = " + ".join(top_words) 
    topicos_nomes.append(f"Tópico {i+1}: {nome_topico}")

# grafico pagina web
st.subheader("Macrotendências Híbridas (BM25 + PageRank + Pesos)")

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(topicos_nomes[::-1], topicos_pesos[::-1], color='darkorange')
ax.set_xlabel('Força Híbrida do Tópico')
ax.set_title('Termos de Impacto Direcionados pelo Pesquisador')
plt.tight_layout()

st.pyplot(fig)