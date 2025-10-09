from libs.Utils.Connection import COLLS, get_coll
from libs.Utils.Exception import Http_Exception
from sentence_transformers import SentenceTransformer

def criar_embedding():
    try:
        cursor = get_coll(COLLS["produto"])

        sem_embedding = cursor.find({"cEmbedding":{"$exists":False}}).to_list()

        if (len(sem_embedding) > 0):
            model_embedding = SentenceTransformer("all-MiniLM-L6-v2")
            for prd in sem_embedding:
                nome_produto = prd["cNmProduto"]
                id_produto = prd["_id"]
                embedding = model_embedding.encode(nome_produto).tolist()

                cursor.update_one({"_id":id_produto}, {"$set":{"cEmbedding":embedding}})

    except Exception as ex:
        raise Http_Exception(500, f"Ocorreu um erro ao realizar o embedding dos produtos.\nErro:{ex}")