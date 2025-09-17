from fastapi import FastAPI
from libs.TableCreator import criar_tabela_nutricional
from libs.Nutr_IA import nutria_bot

api = FastAPI()

@api.get("/")
async def index():
    return{"message":"Bem-vindo ao FastTria! Para acessar a documentação da API entre no swagger no endpoint: /docs#/"}

@api.post("/tablecreator/{cod_user}")
async def create_table(cod_user:int):
    retorno = criar_tabela_nutricional(cod_user)
    return {"message":retorno}

@api.post("/chatbot/")
async def chat_NutrIA(body: dict):
    pergunta = body["cPrompt"]
    resposta = nutria_bot(body["nCdUser"], body["iChat"], pergunta)
    return {"Pergunta": pergunta, "Resposta":resposta}
