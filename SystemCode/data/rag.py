from sqlalchemy import create_engine, inspect

engine = create_engine(
    "mysql+pymysql://root:@127.0.0.1:3306/fleetcareai"
)

inspector = inspect(engine)

print("Database:", engine.dialect.name)
print("Analysis columns:")
print([c["name"] for c in inspector.get_columns("analysis")])