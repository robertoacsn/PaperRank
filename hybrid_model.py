import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.decomposition import TruncatedSVD

def extrair_macrotendencias(df, termos_positivos, termos_negativos, num_topicos=8):
    """
    Motor Matemático do PaperRank Híbrido:
    Aplica BM25, Multiplicadores do Pesquisador e Semântica Latente (LSI/SVD).
    """
    lixo_academico = ['paper', 'presents', 'study', 'results', 'proposed', 'method', 
                      'research', 'based', 'approach', 'article', 'this', 'we', 'show', 'data']
    stop_words_finais = list(ENGLISH_STOP_WORDS.union(lixo_academico))

    # Frequência Bruta
    vectorizer = CountVectorizer(ngram_range=(2, 3), stop_words=stop_words_finais, max_df=0.60, min_df=2)
    tf_matrix = vectorizer.fit_transform(df['Abstract']).toarray()
    feature_names = vectorizer.get_feature_names_out()

    # Matemática BM25
    k1 = 1.5 
    b = 0.75 
    doc_lengths = tf_matrix.sum(axis=1)
    avgdl = doc_lengths.mean()
    N = len(df)
    doc_freqs = (tf_matrix > 0).sum(axis=0)
    idf = np.log(((N - doc_freqs + 0.5) / (doc_freqs + 0.5)) + 1)
    length_penalty = (1 - b + b * (doc_lengths / avgdl)).reshape(-1, 1)
    bm25_matrix = idf * (tf_matrix * (k1 + 1)) / (tf_matrix + k1 * length_penalty)

    # Aplicação dos Pesos do Pesquisador (Camada Híbrida)
    pesos_pesquisador = np.ones(len(feature_names))
    for i, term in enumerate(feature_names):
        if any(pos in term for pos in termos_positivos):
            pesos_pesquisador[i] = 2.0
        elif any(neg in term for neg in termos_negativos):
            pesos_pesquisador[i] = 0.2

    bm25_hibrido = bm25_matrix * pesos_pesquisador
    super_scores = df['Super_Score'].values
    weighted_bm25 = bm25_hibrido * super_scores[:, np.newaxis]

    # Modelagem LSI (Tópicos)
    lsi_model = TruncatedSVD(n_components=num_topicos, random_state=42)
    lsi_matrix = lsi_model.fit_transform(weighted_bm25)

    topicos_nomes = []
    topicos_pesos = lsi_model.singular_values_

    for i, comp in enumerate(lsi_model.components_):
        top_words_idx = comp.argsort()[::-1][:3]
        top_words = [feature_names[idx] for idx in top_words_idx]
        nome_topico = " + ".join(top_words) 
        topicos_nomes.append(f"Tópico {i+1}: {nome_topico}")

    return topicos_nomes, topicos_pesos