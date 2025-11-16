from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, Numeric, TIMESTAMP
from typing import Literal, Optional, List, Dict, Any
from pydantic import BaseModel, Field

Base = declarative_base()

# SQLAlchemy модель (сохраняем для совместимости)
class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True)
    transaction_id = Column(String, nullable=False)
    transaction_timestamp = Column(TIMESTAMP)
    card_id = Column(Integer)
    expiry_date = Column(String)
    issuer_bank_name = Column(String)
    merchant_id = Column(Integer)
    merchant_mcc = Column(Integer)
    mcc_category = Column(String)
    merchant_city = Column(String)
    transaction_type = Column(String)
    transaction_amount_kzt = Column(Numeric)
    original_amount = Column(Numeric, nullable=True)
    transaction_currency = Column(String)
    acquirer_country_iso = Column(String)
    pos_entry_mode = Column(String)
    wallet_type = Column(String)

# Pydantic схемы согласно контракту
MCC_CATEGORIES = Literal[
    "Clothing & Apparel", "Dining & Restaurants", "Electronics & Software",
    "Fuel & Service Stations", "General Retail & Department", 
    "Grocery & Food Markets", "Hobby, Books, Sporting Goods",
    "Home Furnishings & Supplies", "Pharmacies & Health", 
    "Services (Other)", "Travel & Transportation", "Unknown",
    "Utilities & Bill Payments"
]

POS_ENTRY_MODES = Literal["Chip", "QR_Code", "Contactless", "Swipe"]
WALLET_TYPES = Literal["Bank's QR", "Samsung Pay", "Google Pay", "Apple Pay"]
CITIES = Literal["Astana", "Almaty", "Shymkent", "Other"]
TRANSACTION_TYPES = Literal["ATM_WITHDRAWAL", "BILL_PAYMENT", "ECOM", "P2P_IN", "P2P_OUT", "POS", "SALARY"]

class TransactionSchema(BaseModel):
    id: int = Field(description="Primary key")
    transaction_id: str = Field(description="Transaction identifier")
    transaction_timestamp: str = Field(description="Timestamp of transaction")
    card_id: int = Field(description="Card identifier")
    expiry_date: str = Field(description="Card expiry date")
    issuer_bank_name: str = Field(description="Issuer bank name")
    merchant_id: int = Field(description="Merchant identifier")
    merchant_mcc: int = Field(description="Merchant MCC code")
    mcc_category: MCC_CATEGORIES = Field(description="MCC category")
    merchant_city: CITIES = Field(description="Merchant city")
    transaction_type: TRANSACTION_TYPES = Field(description="Type of transaction")
    transaction_amount_kzt: float = Field(description="Amount in KZT")
    original_amount: float = Field(description="Original amount")
    transaction_currency: str = Field(description="Currency in ISO format")
    acquirer_country_iso: str = Field(description="Acquirer ISO country code")
    pos_entry_mode: Optional[POS_ENTRY_MODES] = Field(description="POS entry mode")
    wallet_type: Optional[WALLET_TYPES] = Field(description="Wallet type")

# Контракты запросов/ответов
class UserQuery(BaseModel):
    natural_language_query: str
    user_id: str
    model: Literal["llm", "api"] = "api"

class FormatDecision(BaseModel):
    output_format: Literal["text", "table", "graph", "diagram"]
    confidence_score: float = Field(ge=0, le=1)
    clarification_question: Optional[str] = None
    refined_query: str

class SQLValidation(BaseModel):
    sql_query: str
    is_safe: bool
    matches_intent: bool
    validation_notes: str
    alternative_query: Optional[str] = None

class ExecutionResult(BaseModel):
    data: List[Dict[str, Any]]
    row_count: int
    execution_time_ms: float

class FinalResponse(BaseModel):
    content: str
    output_format: Literal["text", "table", "graph", "diagram"]
    data_preview: Optional[List[Dict]] = None
    metadata: Dict[str, Any]
