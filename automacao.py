import schedule
import time
import subprocess
from datetime import datetime

def acionar_robo():
    agora = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n[{agora}] Iniciando a busca autônoma na API...")
    
    # O comando subprocess faz o Python abrir outro arquivo Python
    caminho_python = r"C:\Python314\python.exe"
    subprocess.run([caminho_python, "search.py"])
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Tarefa concluída! Dados atualizados para o Dashboard.")

# ==========================================
# CONFIGURAÇÃO DO RELÓGIO
# ==========================================
# Escolha a frequência que você quer. Aqui estão exemplos:

# Opção A: Rodar todo dia em um horário específico (ex: 2 da tarde)
schedule.every().day.at("14:00").do(acionar_robo)

# Opção B: Rodar a cada X horas (descomente para usar)
# schedule.every(12).hours.do(acionar_robo)

print("Gerente de Automação iniciado! O robô está dormindo e aguardando a hora certa...")
print("Pode minimizar esta janela e ir tomar um café.")

# Loop infinito que checa o relógio a cada 1 minuto
while True:
    schedule.run_pending()
    time.sleep(60)