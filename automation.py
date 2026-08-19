import schedule
import time
import subprocess
import sys

def executar_pipeline():
    print(f"\n[AGENDADOR] Iniciando rotina do PaperRank em: {time.ctime()}")
    
    # A MÁGICA DA PORTABILIDADE:
    # sys.executable descobre sozinho o caminho do Python em qualquer PC (Windows, Mac ou Linux)
    python_bin = sys.executable 
    
    try:
        # Chama o arquivo search.py usando o mesmo Python que está rodando agora
        subprocess.run([python_bin, "search.py"], check=True)
        print("[AGENDADOR] Rotina concluída com sucesso!\n")
    except subprocess.CalledProcessError as e:
        print(f"[AGENDADOR] Erro crítico na execução do pipeline: {e}")

# ==========================================
# REGRAS DE AGENDAMENTO
# ==========================================
# Exemplo: Roda todos os dias às 02:00 da manhã (Madrugada)
schedule.every().day.at("02:00").do(executar_pipeline)

# Para testes rápidos, você pode descomentar a linha abaixo para rodar a cada 5 minutos:
# schedule.every(5).minutes.do(executar_pipeline)

if __name__ == "__main__":
    print("=====================================================")
    print(" 🤖 ROBÔ DE AUTOMAÇÃO PAPERRANK INICIADO ")
    print(f" 📂 Interpretador Python: {sys.executable}")
    print("=====================================================\n")
    print("Pressione Ctrl+C para encerrar o servidor de automação.")
    
    # Opcional: Roda o pipeline uma vez imediatamente ao ligar o gerente
    executar_pipeline()
    
    # Loop infinito aguardando o horário agendado
    while True:
        schedule.run_pending()
        time.sleep(60) # Checa o relógio a cada 60 segundos