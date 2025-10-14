from libs.Utils.Connection import COLLS, get_coll
from libs.Utils.Exception import Http_Exception
from dotenv import load_dotenv
import google.generativeai as genai
import os


def criar_embedding():

    # Carrega a chave do arquivo .env
    load_dotenv()
    genai.configure(api_key=os.getenv("GOOGLE_GEMINI_API"))
    
    # Escolha o modelo de embedding — o mais recente é o 'text-embedding-004'
    model = "text-embedding-004"

    # Produtos
    try:
        cursor = get_coll(COLLS["produto"])

        sem_embedding = cursor.find({"cEmbedding":{"$exists":False}}).to_list()

        if (len(sem_embedding) > 0):
            for prd in sem_embedding:
                nome_produto = prd["cNmProduto"]
                id_produto = prd["_id"]
                embedding = genai.embed_content(
                    model=model,
                    content=nome_produto
                )["embedding"]

                cursor.update_one({"_id":id_produto}, {"$set":{"cEmbedding":embedding}})

    except Exception as ex:
        raise Http_Exception(500, f"Ocorreu um erro ao realizar o embedding dos produtos.\nErro:{ex}")
    
    # Ingredientes
    try:
        cursor = get_coll(COLLS["produto"])

        sem_embedding = cursor.find({"cEmbedding":{"$exists":False}}).to_list()

        if (len(sem_embedding) > 0):
            for prd in sem_embedding:
                nome_produto = prd["cNmProduto"]
                id_produto = prd["_id"]
                embedding = genai.embed_content(
                    model=model,
                    content=nome_produto
                )["embedding"]
                cursor.update_one({"_id":id_produto}, {"$set":{"cEmbedding":embedding}})

    except Exception as ex:
        raise Http_Exception(500, f"Ocorreu um erro ao realizar o embedding dos produtos.\nErro:{ex}")