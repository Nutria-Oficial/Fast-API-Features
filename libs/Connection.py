from pymongo import MongoClient
from dotenv import load_dotenv
import os
from pymongo.server_api import ServerApi
from urllib.parse import quote_plus
import valkey


load_dotenv() # Obtendo as variáveis seguras

# Constantes ---------------------------------------------
DB_USERNAME = quote_plus(os.getenv("MONGO_USER"))

DB_PASSWORD = quote_plus(os.getenv("MONGO_PWD")) 

DB_URI = f"mongodb+srv://{DB_USERNAME}:{DB_PASSWORD}@nutriamdb.zb8v6ic.mongodb.net/?retryWrites=true&w=majority"

DB_NAME = "NutriaMDB"

# Criando os nomes fixos das colls para caso os nomes mudem, o código se mantenha funcionando mesmo apenas alterando esse local
COLLS = {
    "memoria":"chat",
    "tabela_nutricional":"tabela",
    "ingrediente":"ingrediente",
    "produto":"produto"
}

# Funções de conexão basica com MongoDB ------------------------
def _get_connection():
    return MongoClient(DB_URI, server_api=ServerApi('1'))

def get_coll(coll):
    client = _get_connection()
    NutriaMDB = client[DB_NAME]
    return NutriaMDB[coll]


# Redis Connection
def get_redis():
    valkey_uri = os.getenv("REDIS_URI")
    return valkey.from_url(valkey_uri)