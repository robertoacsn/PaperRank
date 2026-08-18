import networkx as nx

dados_simulados = {
    "Artigo_A": ["Artigo_B", "Artigo_C"],
    "Artigo_B": ["Artigo_C", "Artigo_D"],
    "Artigo_C": [],
    "Artigo_D": ["Artigo_A"]
}

grafo = nx.DiGraph()

for chave, valor in dados_simulados.items():
    for artigo in valor:
        grafo.add_edge(chave, artigo)



ranking = nx.pagerank(grafo, alpha=0.85)

# 5. Exibindo o resultado final ordenado do maior para o menor prestígio
print("--- PRESTÍGIO DOS ARTIGOS (PageRank) ---")
for artigo, pontuacao in sorted(ranking.items(), key=lambda item: item[1], reverse=True):
    print(f"{artigo}: {pontuacao:.4f}")