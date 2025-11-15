from sqlalchemy import select
from sqlalchemy.orm import sessionmaker
from models import Transaction
from config import DATABASE_URL
import pandas as pd

from sqlalchemy import create_engine

engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)

with Session() as session:
    stmt = select(Transaction).where(
        Transaction.date >= '2023-01-01',
        Transaction.date <= '2023-01-31'
    ).limit(100)
    result = session.execute(stmt).scalars().all()
    df = pd.DataFrame([r.__dict__ for r in result])
    df = df.drop(columns=["_sa_instance_state"], errors='ignore')
    print(df)
