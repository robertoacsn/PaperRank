import schedule
import time
import subprocess
import sys

def executar_pipeline():
    print(f"\n[AGENDADOR] Iniciando rotina do PaperRank em: {time.ctime()}")

    python_bin = sys.executable 
    
    try:
        subprocess.run([python_bin, "search.py"], check=True)
        print("[AGENDADOR] Rotina concluída com sucesso!\n")
    except subprocess.CalledProcessError as e:
        print(f"[AGENDADOR] Erro crítico na execução do pipeline: {e}")


# Exemplo: Roda todos os dias às 02:00 da manhã
schedule.every().day.at("02:00").do(executar_pipeline)

# Para testes rapidos, descomente a linha abaixo.
# schedule.every(5).minutes.do(executar_pipeline)

if __name__ == "__main__":
    print("=====================================================")
    print(" 🤖 ROBÔ DE AUTOMAÇÃO PAPERRANK INICIADO ")
    print(f" 📂 Interpretador Python: {sys.executable}")
    print("=====================================================\n")
    print("Pressione Ctrl+C para encerrar o servidor de automação.")
    
    executar_pipeline()
    
    while True:
        schedule.run_pending()
        time.sleep(60) 