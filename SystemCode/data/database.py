"""One PostgreSQL configuration for vehicle facts and pgvector knowledge."""
import os
from sqlalchemy import create_engine
from sqlalchemy.engine import URL, make_url


def create_postgres_engine(username=None, password=None, host=None, port=None, database=None, *, database_url=None):
    explicit_fields = any(value is not None for value in (username, password, host, port, database))
    configured = database_url or (None if explicit_fields else os.getenv("FLEETCARE_DATABASE_URL"))
    if configured:
        try:
            url = make_url(configured)
            if url.get_backend_name() not in {"postgresql", "postgres"}:
                raise ValueError()
            url = url.set(drivername="postgresql+psycopg")
        except Exception:
            raise ValueError("FLEETCARE_DATABASE_URL must be a valid PostgreSQL URL") from None
    else:
        url = URL.create("postgresql+psycopg", username=username or os.getenv("FLEETCARE_DB_USER", "postgres"),
                         password=password if password is not None else os.getenv("FLEETCARE_DB_PASSWORD", ""),
                         host=host or os.getenv("FLEETCARE_DB_HOST", "127.0.0.1"),
                         port=int(port or os.getenv("FLEETCARE_DB_PORT", "5432")),
                         database=database or os.getenv("FLEETCARE_DB_NAME", "fleetcareai"))
    return create_engine(url, pool_pre_ping=True, connect_args={"connect_timeout": 10, "prepare_threshold": None})
