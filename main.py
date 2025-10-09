from fastapi import FastAPI
from fastapi.responses import JSONResponse
from libs.TableCreator import criar_tabela_nutricional
from libs.Nutr_IA import nutria_bot
from libs.ProductEmbedding import criar_embedding
from libs.Utils.Exception import Http_Exception

api = FastAPI()

@api.get("/")
async def index():
    return JSONResponse(content={"message":"Bem-vindo ao FastTria! Para acessar a documentação da API entre no swagger no endpoint: /docs#/"}, status_code=200)

@api.post("/tablecreator/{cod_user}")
async def create_table(cod_user:int):
    try:
        retorno = criar_tabela_nutricional(cod_user)
        return JSONResponse(content={"message":retorno}, status_code=200)
    except Http_Exception as http:
        return JSONResponse(content={"message":http.mensagem}, status_code=http.codigo)
    except Exception as e:
        return JSONResponse(content={"message":e}, status_code=500)


@api.post("/chatbot/")
async def chat_NutrIA(body: dict):
    try:
        pergunta = body["cPrompt"]
        resposta = nutria_bot(body["nCdUser"], body["iChat"], pergunta)
        return JSONResponse(content={"Pergunta": pergunta, "Resposta":resposta}, status_code=200)
    except Http_Exception as http:
        return JSONResponse(content={"message":http.mensagem}, status_code=http.codigo)
    except Exception as e:
        return JSONResponse(content={"message":e}, status_code=500)


@api.post("/embedding/")
async def embedding():
    try:
        criar_embedding()
        return JSONResponse(content={"message":"Os produtos tiveram o embedding realizado com sucesso"}, status_code=200)
    except Http_Exception as http:
        return JSONResponse(content={"message":http.mensagem}, status_code=http.codigo)
    except Exception as e:
        return JSONResponse(content={"message":e}, status_code=500)