from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from config import DATABASE_URL
from models import Transaction

class ReadOnlyDB:
    def __init__(self, batch_size=1_000_000):
        self.engine = create_engine(DATABASE_URL)
        self.Session = sessionmaker(bind=self.engine)
        self.batch_size = batch_size  # лимит батча по умолчанию

    def read_batch(self, offset=0, limit=None):
        """
        Чтение данных батчами по умолчанию из Transaction.
        offset — с какой id начинать
        limit — сколько строк читать, если None — берется batch_size
        """
        limit = limit or self.batch_size
        with self.Session() as session:
            stmt = select(Transaction).order_by(Transaction.id).offset(offset).limit(limit)
            return session.execute(stmt).scalars().all()

    def count(self):
        """Количество записей в таблице"""
        with self.Session() as session:
            return session.query(Transaction).count()

    def execute_select(self, stmt):
        """
        Принимает любое SQLAlchemy select выражение.
        Возвращает генератор батчей по batch_size, чтобы не грузить память.
        """
        with self.Session() as session:
            offset = 0
            while True:
                batch_stmt = stmt.offset(offset).limit(self.batch_size)
                results = session.execute(batch_stmt).scalars().all()
                if not results:
                    break
                yield results
                offset += self.batch_size
