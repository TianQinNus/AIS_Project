from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "FleetCare AI API"
    app_version: str = "1.0.0"
    debug: bool = False

    database_url: str

    cors_origins: str = "http://localhost:3000,http://localhost:5173"

    # Qwen
    dashscope_api_key: str

    qwen_base_url: str 

    llm_model: str = "qwen3.5-flash"

    llm_enable_thinking: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def cors_origin_list(self) -> list[str]:
        return [
            origin.strip()
            for origin in self.cors_origins.split(",")
            if origin.strip()
        ]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()