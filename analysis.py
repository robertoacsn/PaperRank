import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from hybrid_model import extrair_macrotendencias

print("Procurando a última extração do PaperRank...")
arquivos_csv = glob.glob('paperrank_*.csv')

if not arquivos_csv:
    print("ERRO: Nenhum arquivo encontrado. Rode o search.py primeiro!")
    exit()

arquivo_mais_recente = max(arquivos_csv, key=os.path.getctime)
print(f"Lendo o arquivo: {arquivo_mais_recente}\n")

df = pd.read_csv(arquivo_mais_recente, sep=';')
df = df.dropna(subset=['Abstract'])
gold_df = df.head(50).copy()

termos_positivos = ['performance', 'accuracy', 'optimization', 'scalable', 'real time', 'solution', 'framework']
termos_negativos = ['review', 'survey', 'challenges', 'barriers', 'limitations', 'future work']

print(f"Calculando Modelo Híbrido para {len(gold_df)} artigos...\n")

topicos_nomes, topicos_pesos = extrair_macrotendencias(gold_df, termos_positivos, termos_negativos)

print("\n--- OS GRANDES TÓPICOS DO PAPERRANK HÍBRIDO ---")
for nome, peso in zip(topicos_nomes, topicos_pesos):
    print(f"[{peso:.2f}] {nome}")

plt.figure(figsize=(12, 7))
plt.barh(topicos_nomes[::-1], topicos_pesos[::-1], color='darkorange')
plt.xlabel('Relevância Híbrida')
plt.title('PaperRank: Macrotendências Híbridas (Terminal)')
plt.tight_layout()
plt.show()