from typing import Optional # padrao do python
from langchain.tools import tool
from pydantic import BaseModel, Field
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.schema import HumanMessage
from libs.Utils.Exception import Http_Exception
from libs.Utils.Connection import get_coll, COLLS, get_highest_id, get_api_key
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv
import os
import json

# Carrega a chave do arquivo .env
load_dotenv()
my_api_key = os.getenv("GOOGLE_GEMINI_API")


# Escolha do modelo de embedding — o mais recente é o 'text-embedding-004'
model = "text-embedding-004"

# ============================ Memória =========================== 

# Função que cria o objeto de memória da IA
def create_ChatMessageHistory(messages_json:list[dict]) -> ChatMessageHistory:
    # Objeto de memória da IA
    history = ChatMessageHistory()

    # Passando por cada uma das mensagens e adicionando no objeto ChatMessageHistory
    for m in messages_json:
        if m["type"] == "human":
            history.add_user_message(m["content"])
        elif m["type"] == "ai":
            history.add_ai_message(m["content"])
        elif m["type"] == "system":
            history.add_system_message(m["content"])
   
    return history

# Função que busca a memória da IA
def get_history(nCdUser:int, iChat:int = 1) -> ChatMessageHistory:
    try:
        # Obtendo cursor que interage com o banco de dados
        cursor = get_coll(COLLS["memoria"])

        # Criando a agregação e filtros
        agg = [{"$match": {"nCdUsuario": nCdUser, "iChat":iChat}}, {"$project":{"_id":0, "lMemoria":1}}]

        result = cursor.aggregate(agg).to_list()

        # Caso possua memória
        if (len(result) > 0):
            memoria = create_ChatMessageHistory(result[0]["lMemoria"])
            return memoria
        
        # Caso não possua vai retornar apenas um objeto de memória novo
        return ChatMessageHistory()

    except Exception as ex:
        erro = f"Não foi possível obter a memória do chat do usuário {nCdUser}.\nErro: {ex}"
        raise Http_Exception(500, erro)

# Função que insere a memória da IA dentro do MongoDB
def set_history(nCdUsuario:int, history:ChatMessageHistory, iChat:int=1 ):
    # Obtendo cursor que interage com o banco de dados
    cursor = get_coll(COLLS["memoria"])

    # Tornando a memória em objetos que poderão ser colocados dentro do banco
    lMemoria = [msg.model_dump() if not isinstance(msg, str) else HumanMessage(msg).model_dump() for msg in history.messages]
    lUser = []
    lBot = []

    for memo in lMemoria:
        # print(f"\n\nMemo: {memo}")
        conteudo = str(memo["content"])
        if (memo["type"] == "human"):
            # Regra para adicionar quando for apenas texto
            if ((not conteudo.startswith("{\"route\":")) and (not "\"dominio\":" in conteudo) and (not conteudo.startswith("{\"legal\":")) and (not conteudo.startswith("{\"pergunta_original\":"))):
                lUser.append(conteudo)
        elif (memo["type"] == "ai"):
            # Regra para evitar as memórias das conversas entre os agentes
            if (not "\"dominio\":" in conteudo and (not "\"resposta_small_talk\":null" in conteudo) and (not conteudo.startswith("{\"legal\":")) and (not "\"resposta_final\":" in conteudo)):
                # Verificando se é small talk ou orquestrador
                if ("\"resposta_small_talk\":" in conteudo):
                    resposta_json = json.loads(conteudo)
                    lBot.append(resposta_json.get("resposta_small_talk", ""))
                else:
                    lBot.append(conteudo)
                    
    
    # Verificando pré-existência da memória no banco
    memoria = cursor.find({"nCdUsuario":nCdUsuario, "iChat":iChat}).to_list()

    if (len(memoria) > 0):
        # Já existe
        memoria = memoria[0]
        cursor.update_one({"_id":memoria["_id"]}, {"$set": {"lBot": lBot, "lUser": lUser, "lMemoria": lMemoria}})

    else:
        # Não existe ainda

        # Adquirindo o próximo ID que vai ser inserido
        next_id = get_highest_id(cursor)

        # Criando objeto novo que vai ser inserido e inserindo ele dentro da collection
        memoria = {
            "_id":next_id,
            "nCdUsuario":nCdUsuario,
            "iChat":iChat,
            "lUser":lUser,
            "lBot": lBot,
            "lMemoria":lMemoria
        }

        cursor.insert_one(memoria)


# ========================= Funções Extras ======================== 
def get_datetime(timezone:str = "America/Sao_Paulo"):
    """
    Obtém a data atual de acordo com um timezone. Default = America/Sao_Paulo
    """
    TZ = ZoneInfo(timezone)
    return datetime.now(TZ).date().isoformat()



# ============================ Tools =============================

# ------------------------- Ingredientes -------------------------

# Consulta
class IngredientsFindArgs(BaseModel):
    cNmIngrediente: Optional[str] = Field(default=None, description="Nome do ingrediente (PT-BR)")
    iLimit: int = Field(default=20, description="Limite de dados que a consulta irá trazer, pode ser mais caso o usuário peça ingredientes sem os parâmetros acima. Por exemplo um ingrediente com maior quantidade de calorias registrado no banco.")

@tool("ingredient_find", args_schema=IngredientsFindArgs)
def ingredient_find(
    cNmIngrediente: Optional[str] = None,
    iLimit: int = 20
):
    """
    Consulta os ingredientes dentro do MongoDB pelo nome e categoria
    """
    try:
        # Obtendo cursor que interage com o banco de dados
        cursor = get_coll(COLLS["ingrediente"])

        # Criando a agregação e filtros
        agg = []
        if (cNmIngrediente):
            genai.configure(api_key=my_api_key)
            
            # Gerando o embedding
            query_emb = genai.embed_content(
                    model=model,
                    content=cNmIngrediente
                )["embedding"]
                        
            agg.extend([{"$vectorSearch": {
                        "index": "vector_index",
                        "path": "cEmbedding",
                        "queryVector": query_emb,
                        "numCandidates": 50,
                        "limit": 3
                    }
                },
            {"$set": {"score": {"$meta": "vectorSearchScore"}}}])

        # Colocando o limit na consulta
        agg.append({"$limit":iLimit})

        resultado = cursor.aggregate(agg).to_list()

        return {"status":"ok", "result":resultado}

    except Exception as ex:
        retorno = {"status":"error", "mesage":ex}
        return retorno

# -------------------------- Produtos ----------------------------

# Consulta
@tool("product_find")
def product_find(
    cNmProduto: Optional[str] = None,
):
    """
    Consulta os produtos dentro do MongoDB pelo nome
    """
    try:
        # Obtendo cursor que interage com o banco de dados
        cursor = get_coll(COLLS["produto"])

        # Criando a agregação e filtros
        
        agg = []
        if (cNmProduto):
            genai.configure(api_key=my_api_key)
            query_emb = genai.embed_content(
                    model=model,
                    content=cNmProduto
             )["embedding"]
            
            agg = [
                {
                    "$vectorSearch": {
                        "index": "vector_index",
                        "path": "cEmbedding",
                        "queryVector": query_emb,
                        "numCandidates": 50,
                        "limit": 50
                    }
                },
                {"$set": {"score": {"$meta": "vectorSearchScore"}}}
            ]

        resultado = cursor.aggregate(agg).to_list()

        return {"status":"ok", "result":resultado}

    except Exception as ex:
        return {"status":"error", "mesage":ex}


# ---------------------- Tabela Nutricional ----------------------

# Consulta
class NutricionalTableFindArgs(BaseModel):
    nCdProduto: Optional[int] = Field(default=None, description="FK de produto ao qual a tabela nutricional pertence")
    cNmProduto: Optional[str] =  Field(default=None, description="Nome do produto a qual a tabea nurtricional pertence (Caso esteja inserindo uma tabela nutricional é obrigatório)")
    cNmTabela: Optional[str] = Field(default=None, description="Nome da tabela nutricional, filtro de texto (Caso esteja inserindo uma tabela nutricional é obrigatório)")
    cUnidadeMedida: Optional[str] = Field(default=None, description="Unidade de medida da tabela nutricional (Caso esteja inserindo uma tabela nutricional é obrigatório)")

@tool("table_find", args_schema=NutricionalTableFindArgs)
def table_find(
    nCdProduto: Optional[int] = None,
    cNmProduto: Optional[str] = None,
    cNmTabela: Optional[str] = None,
    cUnidadeMedida: Optional[str] = None
):
    """
    Tool responsável por buscar todas tabelas nutricionais de acordo com filtros especificados pelo usuário
    """
    try:
        # Obtendo cursor que interage com o banco de dados
        cursor = get_coll(COLLS["tabela_nutricional"])

        # Criando a agregação e filtros
        agg = [{"$match": {"_id": {"$exists": True}}}]

        if (cNmTabela):
            agg[0]["$match"]["cNmTabela"] = cNmTabela

        if (cUnidadeMedida):
            agg[0]["$match"]["cUnidadeMedida"] = cUnidadeMedida

        # Adicionando o lookup dentro da pipeline
        agg.extend([
        {"$lookup": {
            "from": "produto", 
            "localField": "nCdProduto", 
            "foreignField": "_id", 
            "as": "cNmProduto"
            }}, 
        {"$unwind": "$cNmProduto"}, 
        {"$set": {"cNmProduto": "$cNmProduto.cNmProduto"}}
        ])

        # Filtros pelo produto
        if (nCdProduto):
            agg.append({"$match": {"nCdProduto": nCdProduto}})

        elif (cNmProduto):
            agg.append({"$match": {"cNmProduto": cNmProduto}})

        resultado = cursor.aggregate(agg).to_list()

        return {"status":"ok", "result":resultado}

    except Exception as ex:
        return {"status":"error", "mesage":ex}        


# Inserção
class NutricionalTableInsertArgs(NutricionalTableFindArgs):
    nPorcao: float = Field(default=..., description="Quantidade de (unidade de medida) presente em cada uma das porções da tabela nutricional")
    lIngredientes: list[dict] = Field(default=..., description="Lista de dicionários que contém as chaves [cNmIngrediente, iQuantidade]")

@tool("table_insert", args_schema=NutricionalTableInsertArgs)
def table_insert(
    nPorcao: float,
    lIngredientes: list[dict],
    nCdProduto: Optional[int] = None,
    cNmProduto: Optional[str] = None,
    cNmTabela: Optional[str] = None,
    cUnidadeMedida: Optional[str] = None
):
    """
    Tool que pega a receita de um alimento para gerar sua tabela nutricional
    """
    try:
        # Importando a biblioteca apenas quando for criar uma tabela nutricional para aumentar a velocidade das outras respostas
        from libs.TableCreator import criar_tabela_nutricional_IA

        # Obtendo cursor que interage com o banco de dados para pegar os códigos dos ingredientes
        cursor = get_coll(COLLS["ingrediente"])

        for i in lIngredientes:
            nome_ingrediente = i.pop("cNmIngrediente")
            genai.configure(api_key=my_api_key)
            query_emb = genai.embed_content(
                    model=model,
                    content=nome_ingrediente
                )["embedding"]
            agg = [
                {
                    "$vectorSearch": {
                        "index": "vector_index",
                        "path": "cEmbedding",
                        "queryVector": query_emb,
                        "numCandidates": 50,
                        "limit": 1
                    }
                },
                {"$set": {"score": {"$meta": "vectorSearchScore"}}}
            ]

            result = cursor.aggregate(agg).to_list()[0]

            if (result):
                i["nCdIngrediente"] = int(result["_id"])
            else:
                raise Exception(f"Não foi possível encontrar o ingrediente com nome {nome_ingrediente}")
        
            
        if (not nCdProduto):
            # Mudando o cursor para conectar na collection de produtos e buscar o código do produto
            cursor = get_coll(COLLS["produto"])
            
            genai.configure(api_key=my_api_key)
            query_emb = genai.embed_content(
                    model=model,
                    content=cNmProduto
             )["embedding"]
            
            agg = [
                {
                    "$vectorSearch": {
                        "index": "vector_index",
                        "path": "cEmbedding",
                        "queryVector": query_emb,
                        "numCandidates": 50,
                        "limit": 1
                    }
                },
                {"$set": {"score": {"$meta": "vectorSearchScore"}}}
            ]

            result = cursor.aggregate(agg).to_list()[0]

            if (result):
                nCdProduto = int(result["_id"])
            else: 
                raise Exception(f"Não foi possível encontrar o produto com nome {cNmProduto}")
            

        # Criando tabela nutricional e colocando no banco
        result = criar_tabela_nutricional_IA(cNmTabela, nPorcao, lIngredientes, cUnidadeMedida, nCdProduto)

        return {"status":"ok", "mesage":result}

    except Exception as ex:
        return {"status":"error", "mesage":ex}     


# --------------------------- App -------------------------------
@tool("search_fluxo")
def search_fluxo():
    """
    Tool responsável por obter os dados sobre o fluxo do app
    """
    try:
        texto = ""

        # Buscando o caminho relacionado ao arquivo que contém o fluxo do app
        caminho_fluxo = Path(__file__).resolve().parent.parent / "docs" / "app" / "fluxo_nutria.txt"
        
        # Abrindo o arquivo, pegando o conteúdo e retornando
        with(open(caminho_fluxo, encoding="utf-8")) as fluxo:
            texto = fluxo.read()
        return {"status":"ok", "fluxo":texto}
    except Exception as ex:
        return {"status":"error", "error":ex}

# ------------------ Variáveis importaveis ----------------------
MEMORY = [get_history, set_history]
FUNCTIONS = [get_datetime]
TOOLS_BD = [ingredient_find, product_find, table_find, table_insert]
TOOLS_RAG = [search_fluxo]