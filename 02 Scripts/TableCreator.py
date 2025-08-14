# Importações necessárias
import os
import pandas as pd
import json
from pymongo import MongoClient
from urllib.parse import quote_plus
import valkey
from dotenv import load_dotenv

__all__ = ["criar_tabela_nutricional"]

load_dotenv() # Obtendo as variáveis seguras

# Conexões

# MongoDB
username = quote_plus(os.getenv("MONGO_USER"))
password = quote_plus(os.getenv("MONGO_PWD")) 

uri = f"mongodb+srv://{username}:{password}@nutriamdb.zb8v6ic.mongodb.net/?retryWrites=true&w=majority"

conn = MongoClient(uri)
db = conn["NutriaMDB"]
coll_ingrediente = db["ingrediente"]
coll_tabela = db["tabela"]


# Redis
valkey_uri = os.getenv("REDIS_URI")
redis = valkey.from_url(valkey_uri)
prefixo_requisicao_user = "requisicao_user:"

# Constantes
nome_ptbr = {
    "nCaloria(kcal)":"Valor Calórico (kcal)",
    "nProteina(g)":"Proteína (g)",
    "nCarboidrato(g)":"Carboidrato (g)",
    "nAcucar(g)":"Açúcar Total (g)",
    "nFibra(g)":"Fibra Alimentar (g)",
    "nGorduraTotal(g)":"Gordura Total (g)",
    "nGorduraSaturada(g)":"Gordura Saturada (g)",
    "nGorduraMonoinsaturada(g)":"Gordura Monoinsaturada (g)",
    "nGorduraPoliinsaturada(g)":"Gordura Poli-Insaturada (g)",
    "nColesterol(mg)":"Colesterol (mg)",
    "nRetinol(mcg)":"Retinol/Vitamina A (μg)",
    "nTiamina(mg)":"Tiamina (mg)",
    "nRiboflavina(mg)":"Riboflavina (mg)",
    "nNiacina(mg)":"Niacina (mg)",
    "nVitB6(mg)":"Vitamina B-6 (mg)",
    "nFolato(mcg)":"Ácido Fólico (μg)",
    "nColina(mg)":"Colina (mg)",
    "nVitB12(mcg)":"Vitamina B-12 (μg)",
    "nVitC(mg)":"Vitamina C (mg)",
    "nVitD(mcg)":"Vitamina D (μg)",
    "nVitE(mg)":"Vitamina E (mg)",
    "nVitK(mcg)":"Vitamina K (μg)",
    "nCalcio(mg)":"Cálcio (mg)",
    "nFosforo(mg)":"Fósforo (mg)",
    "nMagnesio(mg)":"Magnésio (mg)",
    "nFerro(mg)":"Ferro (mg)",
    "nZinco(mg)":"Zinco (mg)",
    "nCobre(mg)":"Cobre (mg)",
    "nSelenio(mcg)":"Selênio (μg)",
    "nPotassio(mg)":"Potássio (mg)",
    "nSodio(mg)":"Sódio (mg)",
    "nCafeina(mg)":"Cafeína (mg)",
    "nTeobromina(mg)":"Teobromina (mg)",
    "nAlcool(g)":"Álcool (g)",
    "nAgua(g)":"Água (g)",
}

vd_referencia = {
    "nCaloria(kcal)":2000,
    "nProteina(g)":50,
    "nCarboidrato(g)":300,
    "nAcucar(g)":50,
    "nFibra(g)":25,
    "nGorduraTotal(g)":65,
    "nGorduraSaturada(g)":20,
    "nGorduraMonoinsaturada(g)":20,
    "nGorduraPoliinsaturada(g)":20,
    "nColesterol(mg)":300,
    "nRetinol(mcg)":800,
    "nTiamina(mg)":1.2,
    "nRiboflavina(mg)":1.2,
    "nNiacina(mg)":15,
    "nVitB6(mg)":1.3,
    "nFolato(mcg)":400,
    "nColina(mg)":550,
    "nVitB12(mcg)":2.4,
    "nVitC(mg)":100,
    "nVitD(mcg)":15,
    "nVitE(mg)":15,
    "nVitK(mcg)":120,
    "nCalcio(mg)":1000,
    "nFosforo(mg)":700,
    "nMagnesio(mg)":120,
    "nFerro(mg)":14,
    "nZinco(mg)":11,
    "nCobre(mg)":0.9,
    "nSelenio(mcg)":60,
    "nPotassio(mg)":3500,
    "nSodio(mg)":2000,
    "nCafeina(mg)":0,
    "nTeobromina(mg)":0,
    "nAlcool(g)":0,
    "nAgua(g)":0,
}


# Funções ---------------------------------

def __gerar_tabela_nutricional(ingredientes:list[dict], porcao:float) -> pd.DataFrame:
    """
    ## Recebe
    ### Lista de ingredientes em formato ``code`` e ``amount``
    Uma lista de dicinários com esse formato, onde code é o código do ingrediente especificado na tabela principal, e amount é a quantidade desse ingrediente por porção

    ## Retorna uma lista com um:
    ### ``DataFrame`` com as informações da tabela nutricional
    Uma tabela com as colunas:
    - ``Nutriente``: Nome da nutriente (seja ele caloria ou vitamina) em PT-BR
    - ``100g``: Quantidade daquela nutriente em 100g
    - ``200g``: Quantidade daquela nutriente em 200g
    - ``VD`` : Porcentagem daquela nutriente no contexto geral da tabela

    ### O `total` que contém a quantidade total de volume/peso que a tabela contém
    
    ### E os `ingredientes` que contém uma lista de dicionarios com os ingredientes usados na tabela e suas quantidades
    """
    
    # Informações da tabela
    table_info = {
        "nCaloria(kcal)":0,
        "nProteina(g)":0,
        "nCarboidrato(g)":0,
        "nAcucar(g)":0,
        "nFibra(g)":0,
        "nGorduraTotal(g)":0,
        "nGorduraSaturada(g)":0,
        "nGorduraMonoinsaturada(g)":0,
        "nGorduraPoliinsaturada(g)":0,
        "nColesterol(mg)":0,
        "nRetinol(mcg)":0,
        "nTiamina(mg)":0,
        "nRiboflavina(mg)":0,
        "nNiacina(mg)":0,
        "nVitB6(mg)":0,
        "nFolato(mcg)":0,
        "nColina(mg)":0,
        "nVitB12(mcg)":0,
        "nVitC(mg)":0,
        "nVitD(mcg)":0,
        "nVitE(mg)":0,
        "nVitK(mcg)":0,
        "nCalcio(mg)":0,
        "nFosforo(mg)":0,
        "nMagnesio(mg)":0,
        "nFerro(mg)":0,
        "nZinco(mg)":0,
        "nCobre(mg)":0,
        "nSelenio(mcg)":0,
        "nPotassio(mg)":0,
        "nSodio(mg)":0,
        "nCafeina(mg)":0,
        "nTeobromina(mg)":0,
        "nAlcool(g)":0,
        "nAgua(g)":0,
    }

    total_amount = 0

    # Percorrendo os ingredientes para adicionar as informações
    for ingrediente in ingredientes:
        total_amount += float(ingrediente["iQuantidade"])
        ingrediente_code = int(ingrediente["nCdIngrediente"])

        row = coll_ingrediente.aggregate([{"$match":{"_id":ingrediente_code}},
                                         {"$project":{"_id":0, "cNmIngrediente":0, "cCategoria":0}}]).to_list()[0]

        for key, value in row.items():
            table_info[key] += value

    nutrientes_info = {
        "cNutriente":[nome_ptbr[key] for key in table_info.keys()], 
        "nTotal":[value for value in table_info.values()],
        "nPorcao":[value/total_amount*porcao for value in table_info.values()],
        "nVD":[value/total_amount*porcao/vd_referencia[key]*100 if vd_referencia[key] != 0 else None for key, value in table_info.items()]
        }
    
    df_final = pd.DataFrame(nutrientes_info)

    return df_final,total_amount

def __inserir_tabela_bd(cod_user:int, nome_tabela:str, total_tabela:float, porcao:float, ingredientes:list[dict], tabela:dict):

    try:
        aggregate = [{"$sort":{"_id":-1}},
        {"$limit":1},
        {"$project":{"_id":1}}]

        next_id = coll_tabela.aggregate(aggregate).to_list()[0]["_id"]+1

        cod_produto = float(redis.hget(prefixo_requisicao_user+str(cod_user), "cod_produto"))

        tabela_banco = {
        "_id":next_id,
        "nCdProduto":cod_produto,
        "cNmTabela":nome_tabela,
        "nTotal":total_tabela,
        "nPorcao":porcao,
        "lIngredientes":ingredientes,
        "lNutrientes":tabela["cNutriente"],
        "lTotal":tabela["nTotal"],
        "lPorcao":tabela["nPorcao"],
        "lVd":tabela["nVD"]
        }

        coll_tabela.insert_one(tabela_banco)

        print(f"A tabela {nome_tabela} foi inserida com sucesso")

    except Exception as e:
        print("Ocorreu um erro ao inserir no banco de dados")
        print("Erro: \n"+str(e))


# ----------------------------------------
# Obtendo informações do Redis
# ----------------------------------------


def criar_tabela_nutricional(cod_user:int):
    try:
        # Pegando parâmetros passados pelo Redis
        nome_tabela = str(redis.hget(prefixo_requisicao_user+str(cod_user), "nome_tabela"))
        nome_tabela = nome_tabela.removeprefix("b'").removesuffix("'")
        print(nome_tabela)

        porcao = float(redis.hget(prefixo_requisicao_user+str(cod_user), "porcao_tabela"))
        print(porcao)

        ingredientes_redis = redis.hget(prefixo_requisicao_user+str(cod_user), "ingredientes")

        ingredientes = json.loads(ingredientes_redis.decode("utf-8"))


        # Gerando a tabela nutricional
        tabela, total_tabela = __gerar_tabela_nutricional(ingredientes, porcao)

        tabela = tabela.to_dict('list')

        # Inserindo ela no MongoDB
        __inserir_tabela_bd(cod_user, nome_tabela, total_tabela, porcao, ingredientes, tabela)

        print(f"Tabela nutricional do usuário {cod_user} foi inserida no MongoDB")

    except Exception as e:
        print(f"Ocorreu um erro ao tentar inserir a tabela nutricional do usuário {cod_user}")
        print(f"Erro: {e}")