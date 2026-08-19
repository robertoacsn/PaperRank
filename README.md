# PaperRank: Pipeline Autônomo de Inteligência Tecnológica

[![Status](https://img.shields.io/badge/Status-Em%20Desenvolvimento-yellow)]()

O **PaperRank** é um sistema projetado para automatizar a revisão bibliográfica e a descoberta de macrotendências tecnológicas. Ele atua como um pipeline de dados que integra a extração de metadados acadêmicos, o processamento de linguagem natural e a análise de redes, exibindo os resultados em um painel interativo. O objetivo principal é eliminar o viés manual na pesquisa e identificar agrupamentos semânticos emergentes por meio de algoritmos não-supervisionados de Machine Learning.

## Funcionalidades Principais

*   **Extração de Dados:** Consome a API REST do *Semantic Scholar* (Graph API v1) para capturar metadados e resumos (abstracts) de artigos científicos.
*   **Recuperação da Informação e Modelagem de Tópicos:** Implementa o motor matemático **Okapi BM25** e a técnica de **SVD** (Singular Value Decomposition) pelo método **LSI** (Latent Semantic Indexing) para identificar os principais tópicos e agrupamentos semânticos na literatura.
*   **Métrica Híbrida de Ranqueamento:** Calcula um `Super_Score` baseando-se na **Teoria dos Grafos** (PageRank aplicado à rede de referências) e na velocidade de citação dos documentos.
*   **Dashboard Interativo:** Uma interface web construída com **Streamlit** que exibe uma visão tabular dos artigos ranqueados e gráficos de barras dos tópicos semânticos mais fortes.
*   **Exportação de Dados:** Exporta a base de dados consolidada e tratada para o formato plano (.csv).
*   **Tolerância a Falhas:** Rotinas de *retry exponencial* ("backoff") para assegurar estabilidade contra limites de requisição (Rate Limits) ou indisponibilidades temporárias da API.

## Tecnologias e Arquitetura

*   **Linguagem Base:** Python 3.x
*   **Interface (Frontend):** Streamlit
*   **Integração:** HTTP REST API (Semantic Scholar)
*   **Processamento/Matemática:** Numpy, algoritmos focados em NLP (BM25, LSI) e Grafos (PageRank).
*   **Armazenamento de Dados:** Manipulação de arquivos locais (.csv e .txt), dispensando banco de dados relacional nesta fase.

## Como funciona o Pipeline

1.  **Ingestão:** O sistema faz requisições protegidas (com tolerância a falhas) à API do Semantic Scholar, coletando resumos e informações de citação.
2.  **Processamento e Avaliação (Back-end):**
    *   Os textos passam pelo modelo de Retrieval Information (BM25 + LSI).
    *   O algoritmo avalia a relevância de cada documento aplicando o algoritmo PageRank sobre as conexões de referência, gerando o `Super_Score`.
3.  **Estruturação:** O sistema mantém o rastreio (em `historico_tendencias.txt`) para evitar loops redundantes e estrutura o *output* em planilhas tabulares.
4.  **Visualização (Front-end):** A interface carrega e renderiza de forma amigável e interativa as tabelas organizáveis e os gráficos baseados na pontuação semântica extraída.

## Público-Alvo e Requisitos de Execução

Projetado para cientistas de dados, engenheiros, pesquisadores acadêmicos e analistas de inovação. A execução do código requer familiaridade básica com o terminal para inicializar os scripts (ex: `streamlit run app.py`), mas a exploração dos dados pelo Dashboard independe de conhecimentos técnicos prévios.

## Restrições Atuais do Projeto

*   **Fonte de Textos:** Análise circunscrita aos resumos (*abstracts*), não consumindo o texto integral dos artigos (full-text).
*   **Limite de Requisições:** A extração submete-se ao Rate Limit gratuito da Graph API (aprox. 10 requests/segundo).
*   **Ambiente:** Projetado primeiramente para execução em ambiente local (*Localhost*).

## Histórico de Versão

*   **13/08/2026 - v1.0:** Criação do Documento de Especificação de Requisitos e definição da arquitetura base pelo autor (Roberto Almeida Corrêa dos Santos Neto).
