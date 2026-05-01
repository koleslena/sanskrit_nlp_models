import os
import csv
import subprocess
from datetime import datetime
from github import Github, Auth 
from dotenv import load_dotenv

from config import Config

load_dotenv()

def push_to_hub(model_path):
    token = os.getenv("GITHUB_TOKEN")

    # Загружаем модель в Releases через API (не забивая репозиторий)
    try:
        # Теперь создаем объект авторизации 
        auth = Auth.Token(token) 
        
        # Передаем объект auth 
        g = Github(auth=auth)
        repo = g.get_repo(Config.repo_name)
        tag = f"model-{datetime.now().strftime('%Y%m%d-%H%M')}"
        
        # Создаем релиз
        release = repo.create_git_release(tag, tag, f"Auto-save from server")
        
        # Загружаем тяжелый файл модели
        print(f"--- Uploading asset to {tag}... ---")
        asset = release.upload_asset(model_path)
        model_url = asset.browser_download_url # Прямая ссылка на скачивание
        print(f"--- Model uploaded to Release: {tag} ---")

        # Записываем данные в releases.csv
        csv_releases_file = "releases.csv"
        file_exists = os.path.isfile(csv_releases_file)
        
        row = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "tag": tag,
            "url": model_url
        }
        
        with open(csv_releases_file, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader() 
            writer.writerow(row)
            
        print(f"--- Data appended to {csv_releases_file} ---")

        # Пушим код и логи как обычно
        subprocess.run(["git", "add", "metrics", f"{csv_releases_file}"], check=True) # Добавляем только легкие файлы
        commit_msg = f"Training log: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        subprocess.run(["git", "commit", "-m", commit_msg], check=True)
        subprocess.run(["git", "push", "origin", "main"], check=True)

    except Exception as e:
        print(f"Release upload failed: {e}")

if __name__ == "__main__":
    push_to_hub("segmenter_output/segmenter_model.pth")