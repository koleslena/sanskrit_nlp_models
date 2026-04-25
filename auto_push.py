import subprocess
from datetime import datetime

def git_push_results():
    try:
        # 1. Добавляем все новые файлы (модели, логи, признаки)
        subprocess.run(["git", "add", "."], check=True)
        
        # 2. Формируем сообщение коммита с датой и временем
        commit_msg = f"Auto-save results: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        subprocess.run(["git", "commit", "-m", commit_msg], check=True)
        
        # 3. Пушим в репозиторий
        subprocess.run(["git", "push", "origin", "main"], check=True)
        print("--- Results successfully pushed to GitHub! ---")
        
    except Exception as e:
        print(f"--- Git push failed: {e} ---")

if __name__ == "__main__":
    git_push_results()
