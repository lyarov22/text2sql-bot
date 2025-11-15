import json
from google import genai
from google.genai import types

from app.config import LLM_API_KEY

client = genai.Client(api_key=LLM_API_KEY)

TABLE_SCHEMA = {
    "id": "Integer primary key",
    "transaction_id": "String transaction identifier",
    "transaction_timestamp": "Timestamp of transaction",
    "card_id": "Integer card identifier",
    "expiry_date": "String card expiry date",
    "issuer_bank_name": "String issuer bank name",
    "merchant_id": "Integer merchant identifier",
    "merchant_mcc": "Integer merchant mcc code",
    "mcc_category": "String mcc category. Only this values: (Clothing & Apparel, Dining & Restaurants, Electronics & Software, Fuel & Service Stations, General Retail & Department, Grocery & Food Markets, Hobby, Books, Sporting Goods, Home Furnishings & Supplies, Pharmacies & Health, Services (Other), Travel & Transportation, Unknown, Utilities & Bill Payments)",
    "merchant_city": "String merchant city. Example: (Astana, Almaty, Shymkent, Other)",
    "transaction_type": "String type of transaction. Example: (ATM_WITHDRAWAL, BILL_PAYMENT, ECOM, P2P_IN, P2P_OUT, POS, SALARY)",
    "transaction_amount_kzt": "Numeric amount in KZT",
    "original_amount": "Numeric original amount",
    "transaction_currency": "String currency in ISO format. Example: (ARM, BLR, CHN, GEO, ITA, KAZ, KGZ, TUR, USA, UZB)",
    "acquirer_country_iso": "String acquirer ISO. Example: ()",
    "pos_entry_mode": "String pos entry mode. Only this values: (Chip, QR_Code, Contactless, Swipe)",
    "wallet_type": "String wallet type. Only this values: (Bank's QR, Samsung Pay, Google Pay, Apple Pay)"
}

DEFAULT_LIMIT = 100000

SYSTEM_PROMPT = f"""
You are an enterprise Text2SQL generator. Output only SQL. Use table 'transactions'. Use lowercase column names.
All data in the database is stored in English. For example, city names, bank names, MCC categories, transaction types, currencies are all in English.
If the user speaks Russian or Kazakh, translate their intent to English values where appropriate. 
Always output SQL in English syntax and using English values.

If listing rows and no limit is given, apply LIMIT {DEFAULT_LIMIT}.
Allowed operations: SELECT, WHERE, GROUP BY, ORDER BY, LIMIT, AVG, SUM, COUNT, MIN, MAX.
Dates filtered through transaction_timestamp.
Amounts filtered through transaction_amount_kzt.
String filters must use ILIKE.
Never explain anything. Never output text other than SQL.
Return only raw SQL, starts at SELECT, without "```sql".

Table schema:
{json.dumps(TABLE_SCHEMA, indent=2)}
"""

ITER_1 = """
Extract intent, target columns, filters, aggregation, ordering, limits, and time range.
Return JSON:
{
  "aggregation": "... or null",
  "target_columns": [...],
  "filters": [...],
  "order": "... or null",
  "limit": number or null,
  "time_range": { "from": "...", "to": "..." } or null
}
Return only JSON.
"""

ITER_2 = """
Using the JSON, build a correct PostgreSQL SQL query.
Only SQL. Lowercase columns. Use ILIKE for string filters.
Return only raw SQL, starts at SELECT, without "```sql".
"""


class Text2SQLGenerator:
    def __init__(self):
        self.model = "gemini-2.5-flash"

    def _call(self, system_instruction: str, user_text: str) -> str:
        contents = types.Content(
            role="user",
            parts=[types.Part.from_text(text=system_instruction), types.Part.from_text(text=user_text)]
        )
        config = types.GenerateContentConfig(
            system_instruction=None,  # system instruction is passed via content parts
            temperature=0.0,
            max_output_tokens=1500
        )
        response = client.models.generate_content(
            model=self.model,
            contents=[contents],
            config=config
        )
        print(response)
        return response.text

    def generate(self, nl_query: str) -> str:
        step1 = self._call(SYSTEM_PROMPT, ITER_1 + "\n" + nl_query)
        print(step1)
        step2 = self._call(SYSTEM_PROMPT, ITER_2 + "\n" + step1)
        print(step2)
        return step2

def build_text2sql():
    return Text2SQLGenerator()