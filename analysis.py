import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.decomposition import TruncatedSVD

# pega csv mais recente
print("Procurando a última extração do PaperRank...")
arquivos_csv = glob.glob('paperrank_*.csv')

if not arquivos_csv:
    print("ERRO: Nenhum arquivo 'paperrank_*.csv' encontrado. Rode o search.py primeiro!")
    exit()

arquivo_mais_recente = max(arquivos_csv, key=os.path.getctime)
print(f"Lendo o arquivo: {arquivo_mais_recente}\n")

df = pd.read_csv(arquivo_mais_recente, sep=';')
df = df.dropna(subset=['Abstract'])
gold_df = df.head(50).copy()

print(f"Calculando Modelo Híbrido (BM25 + Grafos + Pesos do Pesquisador) para {len(gold_df)} artigos...\n")


# A CAMADA DO PESQUISADOR (Critérios de Priorização)
termos_positivos = ['performance', 'accuracy', 'optimization', 'scalable', 'real time', 'solution', 'framework']
termos_negativos = ['review', 'survey', 'challenges', 'barriers', 'limitations', 'future work']

lixo_academico = ['paper', 'presents', 'study', 'results', 'proposed', 'method', 'research', 'based', 'approach', 'article', 'this', 'we', 'show', 'data']
stop_words_finais = list(ENGLISH_STOP_WORDS.union(lixo_academico))


# EXTRAÇÃO TF E MATEMÁTICA DO BM25

vectorizer = CountVectorizer(ngram_range=(2, 3), stop_words=stop_words_finais, max_df=0.60, min_df=2)
tf_matrix = vectorizer.fit_transform(gold_df['Abstract']).toarray()
feature_names = vectorizer.get_feature_names_out()

k1 = 1.5 
b = 0.75 

doc_lengths = tf_matrix.sum(axis=1)
avgdl = doc_lengths.mean()
N = len(gold_df)
doc_freqs = (tf_matrix > 0).sum(axis=0)
idf = np.log(((N - doc_freqs + 0.5) / (doc_freqs + 0.5)) + 1)

length_penalty = (1 - b + b * (doc_lengths / avgdl)).reshape(-1, 1)
bm25_matrix = idf * (tf_matrix * (k1 + 1)) / (tf_matrix + k1 * length_penalty)


# APLICAÇÃO DOS PESOS DO PESQUISADOR NA MATRIZ BM25
pesos_pesquisador = np.ones(len(feature_names)) # Todo termo começa com peso 1 (Neutro)

for i, term in enumerate(feature_names):
    # Se uma palavra do pesquisador estiver no termo encontrado
    if any(pos in term for pos in termos_positivos):
        pesos_pesquisador[i] = 2.0  # Dobra a relevância (Multiplicador Positivo)
    elif any(neg in term for neg in termos_negativos):
        pesos_pesquisador[i] = 0.2  # Destrói 80% da relevância (Multiplicador Negativo)

# Multiplica a matriz do BM25 pelos multiplicadores do pesquisador
bm25_hibrido = bm25_matrix * pesos_pesquisador


# CRUZAMENTO DE ALGORITMOS (BM25 Híbrido x PageRank)

super_scores = gold_df['Super_Score'].values
weighted_bm25 = bm25_hibrido * super_scores[:, np.newaxis]

# LSI
print("Aplicando SVD para encontrar as Macrotendências validadas pelo Pesquisador...")
NUM_TOPICOS = 8 
lsi_model = TruncatedSVD(n_components=NUM_TOPICOS, random_state=42)

lsi_matrix = lsi_model.fit_transform(weighted_bm25)
topicos_nomes = []
topicos_pesos = lsi_model.singular_values_

print("\n--- OS GRANDES TÓPICOS DO PAPERRANK HÍBRIDO ---")
for i, comp in enumerate(lsi_model.components_):
    top_words_idx = comp.argsort()[::-1][:3]
    top_words = [feature_names[idx] for idx in top_words_idx]
    
    nome_topico = " + ".join(top_words) 
    topicos_nomes.append(f"Tópico {i+1}: {nome_topico}")
    print(f"Tópico {i+1} (Força {topicos_pesos[i]:.2f}): {nome_topico}")

# grafico final
plt.figure(figsize=(12, 7))
plt.barh(topicos_nomes[::-1], topicos_pesos[::-1], color='darkorange')
plt.xlabel('Relevância (BM25 + PageRank + Pesos do Pesquisador)')
plt.title('PaperRank: Macrotendências Híbridas')
plt.tight_layout()
plt.show()