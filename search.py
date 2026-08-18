import requests
import pandas as pd
import networkx as nx
import time
import os
from datetime import datetime
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# ==========================================
# CONFIGURAÇÕES E CREDENCIAIS
# ==========================================
API_KEY = "s2k-XPslyqqALJIt72WXbB55jvhuaextkUwpZdTAO063" 
BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
DESIRED_FIELDS = "title,abstract,year,references,citationCount"

http = Session()
retries = Retry(total=5, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
http.mount('https://', HTTPAdapter(max_retries=retries))
headers = {"x-api-key": API_KEY}

# ==========================================
# FUNÇÕES DO PIPELINE
# ==========================================
def extrair_artigos_da_api(query, limit=100):
    """Bate na API e devolve a lista de artigos brutos (Janela 2021-2026)"""
    print(f"Buscando dados para: '{query}'...")
    params = {
        "query": query,
        "fieldsOfStudy": "Computer Science",
        "fields": DESIRED_FIELDS,
        "year": "2021-2026", # <--- CORREÇÃO TEMPORAL AQUI!
        "limit": limit
    }
    
    try:
        response = http.get(BASE_URL, params=params, headers=headers)
        if response.status_code == 200:
            return response.json().get('data') or []
        else:
            print(f"Erro {response.status_code}: {response.text}")
            return []
    except Exception as e:
        print(f"Erro crítico de conexão: {e}")
        return []

def descobrir_tendencia(articles_list):
    """Usa NLP para achar a palavra mais quente, ignorando histórico"""
    abstracts = [a.get('abstract') for a in articles_list if a.get('abstract')]
    if not abstracts:
        return "machine learning"
    
    tendencias_passadas = set()
    if os.path.exists("historico_tendencias.txt"):
        with open("historico_tendencias.txt", "r", encoding="utf-8") as f:
            for linha in f:
                if "Tendência:" in linha:
                    termo_antigo = linha.split("Tendência:")[1].strip()
                    tendencias_passadas.add(termo_antigo)
    
    lixo_academico = ['paper', 'presents', 'study', 'results', 'proposed', 'method', 'research', 'computer', 'science', 'based', 'approach']
    stop_words_finais = list(ENGLISH_STOP_WORDS.union(lixo_academico))
    
    vectorizer = TfidfVectorizer(ngram_range=(2, 3), stop_words=stop_words_finais, max_df=0.8, min_df=2)
    tfidf_matrix = vectorizer.fit_transform(abstracts)
    
    avg_scores = tfidf_matrix.mean(axis=0).A1
    feature_names = vectorizer.get_feature_names_out()
    top_indices = avg_scores.argsort()[::-1]
    
    for idx in top_indices:
        termo_candidato = feature_names[idx]
        if termo_candidato not in tendencias_passadas:
            return termo_candidato
            
    return feature_names[top_indices[0]]

def construir_grafo_e_tabela(articles_list):
    """Calcula o PageRank e gera a tabela Pandas com Velocidade de Citação Real"""
    structured_data = []
    grafo = nx.DiGraph()
    
    for article in articles_list:
        title_raw = article.get('title') or 'No title'
        title = title_raw.replace('\n', ' ').replace('\r', '').replace(';', ',').replace('"', "'")
        
        year = article.get('year', datetime.now().year)
        origem_id = article.get('paperId')
        lista_referencias = article.get('references') or []
        
        abstract_raw = article.get('abstract')
        if abstract_raw:
            abstract = abstract_raw.replace('\n', ' ').replace('\r', '').replace(';', ',').replace('"', "'")
        else:
            abstract = None
            
        citations = article.get('citationCount') or 0
        
        for ref in lista_referencias:
            ref_id = ref.get('paperId')
            if ref_id:
                grafo.add_edge(origem_id, ref_id)
        
        if abstract:
            structured_data.append({
                "Title": title, "Year": year, "paperId": origem_id,
                "Abstract": abstract, "Citations": citations
            })

    pagerank_scores = nx.pagerank(grafo, alpha=0.85) if len(grafo.nodes) > 0 else {}
    
    if not structured_data:
        return pd.DataFrame()

    df = pd.DataFrame(structured_data)
    current_year = datetime.now().year
    
    # Engenharia de Features (Agora com Idade real!)
    df['Article_Age'] = (current_year - df['Year']).clip(lower=1)
    df['Citation_Velocity'] = df['Citations'] / df['Article_Age']
    df['PageRank'] = df['paperId'].map(pagerank_scores).fillna(0)
    
    max_citations = df['Citations'].max() or 1
    max_velocity = df['Citation_Velocity'].max() or 1
    max_pr = df['PageRank'].max() or 1
    
    df['Score_Citations'] = df['Citations'] / max_citations
    df['Score_Velocity'] = df['Citation_Velocity'] / max_velocity
    df['Score_PageRank'] = df['PageRank'] / max_pr
    
    df['Score_Trend'] = (df['Score_Velocity'] * 0.70) + (df['Score_Citations'] * 0.30)
    df['Super_Score'] = (df['Score_PageRank'] * 0.70) + (df['Score_Trend'] * 0.30)
    
    return df.sort_values(by='Super_Score', ascending=False)

# ==========================================
# O CÉREBRO DO ROBÔ
# ==========================================
if __name__ == "__main__":
    print("=== ESTÁGIO 1: O BATEDOR (VISÃO GLOBAL) ===")
    artigos_globais = extrair_artigos_da_api("Computer Science", limit=100)
    
    tendencia = descobrir_tendencia(artigos_globais)
    print(f"\n>> ALERTA DO NLP: O nicho inédito detectado é: '{tendencia.upper()}' <<\n")
    
    time.sleep(2)
    
    print("=== ESTÁGIO 2: O MERGULHO PROFUNDO (NICHO) ===")
    artigos_nicho = extrair_artigos_da_api(tendencia, limit=100)
    df_final = construir_grafo_e_tabela(artigos_nicho)
    
    if not df_final.empty:
        # Geração do Carimbo (ID Único)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        nome_tendencia = tendencia.replace(' ', '_')
        nome_arquivo_csv = f"paperrank_{nome_tendencia}_{timestamp}.csv"
        
        # Salva o arquivo CSV com nome exclusivo daquela execução
        df_final.to_csv(nome_arquivo_csv, index=False, encoding='utf-8-sig', sep=';')
        
        print(f"\nSucesso! Arquivo '{nome_arquivo_csv}' salvo.")
        print(f"Top 1 Artigo: {df_final.iloc[0]['Title'][:70]}...")
        
        with open("historico_tendencias.txt", "a", encoding="utf-8") as arquivo:
            arquivo.write(f"[{timestamp}] | Tendência: {tendencia} | Arquivo: {nome_arquivo_csv}\n")
    else:
        print("Falha: Nenhum dado válido encontrado para gerar a tabela.")