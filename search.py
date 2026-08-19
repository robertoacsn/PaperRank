import requests
import pandas as pd
import networkx as nx
import time
import os
from datetime import datetime
from dotenv import load_dotenv
import json
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# CONFIGURAÇÕES E CREDENCIAIS

load_dotenv()
API_KEY = os.getenv("API_KEY")
BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
DESIRED_FIELDS = "title,abstract,year,references,citationCount"

http = Session()
retries = Retry(total=5, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
http.mount('https://', HTTPAdapter(max_retries=retries))
headers = {"x-api-key": API_KEY}

# FUNÇÕES PRINCIPAIS

def extrair_artigos_da_api(query, limit=100):
    """Bate na API e devolve a lista de artigos brutos (Janela 2021-2026)"""
    print(f"Buscando dados para: '{query}'...")
    params = {
        "query": query,
        "fieldsOfStudy": "Computer Science",
        "fields": DESIRED_FIELDS,
        "year": "2021-2026",
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

def descobrir_tendencia_temporal(articles_list):
    """Descobre a tendência baseada na ACELERAÇÃO (Regressão Linear ao longo dos anos)"""
    
    # 1. Agrupa os resumos por ano
    textos_por_ano = {}
    for a in articles_list:
        ano = a.get('year')
        resumo = a.get('abstract')
        if ano and resumo:
            textos_por_ano.setdefault(ano, []).append(resumo)
            
    anos_disponiveis = sorted(list(textos_por_ano.keys()))
    
    # Se só tiver 1 ano de dados, a Regressão Linear falha. 
    # Fallback: Usamos o TF-IDF clássico para pegar o termo de maior volume daquele ano.
    if len(anos_disponiveis) < 2:
        todos_resumos = [resumo for lista in textos_por_ano.values() for resumo in lista]
        
        if not todos_resumos:
            return "computer science"
            
        lixo_academico = ['paper', 'presents', 'study', 'results', 'proposed', 'method']
        stop_words_finais = list(ENGLISH_STOP_WORDS.union(lixo_academico))
        
        vec_fallback = TfidfVectorizer(ngram_range=(2, 3), stop_words=stop_words_finais, max_df=0.8, min_df=2)
        tfidf_matrix_fallback = vec_fallback.fit_transform(todos_resumos)
        
        avg_scores = tfidf_matrix_fallback.mean(axis=0).A1
        top_indice = avg_scores.argsort()[::-1][0]
        
        return vec_fallback.get_feature_names_out()[top_indice]

    # 2. Prepara o motor de NLP
    lixo_academico = ['paper', 'presents', 'study', 'results', 'proposed', 'method']
    stop_words_finais = list(ENGLISH_STOP_WORDS.union(lixo_academico))
    vectorizer = TfidfVectorizer(ngram_range=(2, 3), stop_words=stop_words_finais, max_df=0.8, min_df=2)
    
    # Treina o vocabulário com TODOS os textos juntos para fixar as colunas
    todos_resumos = [resumo for lista in textos_por_ano.values() for resumo in lista]
    vectorizer.fit(todos_resumos)
    termos = vectorizer.get_feature_names_out()
    
    # 3. Calcula o TF-IDF médio de cada palavra, ano a ano
    matriz_evolucao = [] # Cada linha será um ano, cada coluna um termo
    for ano in anos_disponiveis:
        textos_do_ano = textos_por_ano[ano]
        tfidf_do_ano = vectorizer.transform(textos_do_ano).mean(axis=0).A1
        matriz_evolucao.append(tfidf_do_ano)
        
    matriz_evolucao = np.array(matriz_evolucao) # (Qtd_Anos, Qtd_Termos)
    
    # 4. Regressão Linear
    melhor_termo = None
    maior_aceleracao = -float('inf') # Infinito negativo (Padrão matemático)
    
    eixo_x = np.array(anos_disponiveis)
    
    for i, termo in enumerate(termos):
        eixo_y = matriz_evolucao[:, i] 
        
        if eixo_y[-1] == 0:
            continue
            
        slope, intercept = np.polyfit(eixo_x, eixo_y, 1)
        
        if slope > maior_aceleracao:
            maior_aceleracao = slope
            melhor_termo = termo
            
    # Trava de segurança final caso nenhuma palavra tenha passado pelo filtro
    if melhor_termo is None:
        return termos[0] if len(termos) > 0 else "computer science"
        
    return melhor_termo

def construir_grafo_e_tabela(articles_list, peso_pr=0.70, peso_trend=0.30, peso_vel=0.70, peso_cit=0.30):
    """Constrói a tabela final filtrando nós pendentes e parametrizando os pesos do Super Score"""
    structured_data = []
    grafo = nx.DiGraph()
    
    # PASSO 1: Mapeia todos os IDs de artigos que realmente baixamos (O "Clube VIP")
    ids_validos = {a.get('paperId') for a in articles_list if a.get('paperId')}
    
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
            if ref_id and (ref_id in ids_validos):
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
    
    # Features
    df['Article_Age'] = (current_year - df['Year']).clip(lower=1)
    df['Citation_Velocity'] = df['Citations'] / df['Article_Age']
    df['PageRank'] = df['paperId'].map(pagerank_scores).fillna(0)
    
    max_citations = df['Citations'].max() or 1
    max_velocity = df['Citation_Velocity'].max() or 1
    max_pr = df['PageRank'].max() or 1
    
    # Normalização (Mantendo todos os scores na mesma escala de 0 a 1)
    df['Score_Citations'] = df['Citations'] / max_citations
    df['Score_Velocity'] = df['Citation_Velocity'] / max_velocity
    df['Score_PageRank'] = df['PageRank'] / max_pr
    
    # O cálculo final agora obedece aos parâmetros recebidos na função:
    df['Score_Trend'] = (df['Score_Velocity'] * peso_vel) + (df['Score_Citations'] * peso_cit)
    df['Super_Score'] = (df['Score_PageRank'] * peso_pr) + (df['Score_Trend'] * peso_trend)
    
    return df.sort_values(by='Super_Score', ascending=False)

# ==========================================
# EXECUÇÃO PRINCIPAL
# ==========================================
if __name__ == "__main__":
    print("=== ESTÁGIO 1: O BATEDOR (VISÃO GLOBAL) ===")
    artigos_globais = extrair_artigos_da_api("Computer Science", limit=100)
    
    tendencia = descobrir_tendencia_temporal(artigos_globais)
    print(f"\n>> ALERTA DO NLP: O nicho inédito detectado é: '{tendencia.upper()}' <<\n")
    
    time.sleep(2)
    
    print("=== ESTÁGIO 2: O MERGULHO PROFUNDO (NICHO) ===")
    artigos_nicho = extrair_artigos_da_api(tendencia, limit=100)
    df_final = construir_grafo_e_tabela(artigos_nicho)
    
    if not df_final.empty:
        # Gera arquivo
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        nome_tendencia = tendencia.replace(' ', '_')
        nome_arquivo_csv = f"paperrank_{nome_tendencia}_{timestamp}.csv"
        nome_arquivo_log = f"paperrank_{nome_tendencia}_{timestamp}_parametros.json"
        
        # Salva CSV
        df_final.to_csv(nome_arquivo_csv, index=False, encoding='utf-8-sig', sep=';')

        # Salva o Log de Parâmetros (Rastreabilidade Científica)
        parametros_usados = {
            "data_execucao": timestamp,
            "tendencia_alvo": tendencia,
            "peso_pagerank": 0.70,
            "peso_trend": 0.30,
            "peso_velocidade": 0.70,
            "peso_citacoes": 0.30
        }
        with open(nome_arquivo_log, "w", encoding="utf-8") as f:
            json.dump(parametros_usados, f, indent=4)
            
        print(f"\nSucesso! Arquivo '{nome_arquivo_csv}' salvo.")
        print(f"Top 1 Artigo: {df_final.iloc[0]['Title'][:70]}...")
        
        with open("historico_tendencias.txt", "a", encoding="utf-8") as arquivo:
            arquivo.write(f"[{timestamp}] | Tendência: {tendencia} | Arquivo: {nome_arquivo_csv}\n")
    else:
        print("Falha: Nenhum dado válido encontrado para gerar a tabela.")