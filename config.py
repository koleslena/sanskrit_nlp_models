import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Загружаем переменные из .env файла
load_dotenv()

class Config:
    repo_name = "koleslena/sanskrit_nlp_models"

class Settings(BaseSettings):
    
    SANSKRIT_TEXTS_DIR: str = "texts/"
    GITHUB_TOKEN: str = "ghp_..."

    # Конфигурация Pydantic
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        extra="ignore" # игнорировать лишние переменные в .env
    )

# Создаем один экземпляр настроек для всего приложения
settings = Settings()