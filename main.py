from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from libs.TableCreator import criar_tabela_nutricional
from libs.TrIA import Tria
from libs.AutomaticEmbedding import criar_embedding
from libs.Utils.Exception import Http_Exception
from libs.TableScanner import processar_imagem
from PIL import Image
import io

api = FastAPI()

# Defina as origens permitidas
origins = [
    "http://localhost",
    "http://localhost:5173",  
    "https://area-restrita-krae.onrender.com",
    # Pode adicionar outras origens se necessário
]

api.add_middleware(
    CORSMiddleware,
    allow_origins=origins,              # Lista de origens permitidas
    allow_credentials=True,             # Permite cookies/credenciais
    allow_methods=["*"],                # Permite todos os métodos (GET, POST, etc.)
    allow_headers=["*"],                # Permite todos os cabeçalhos
)

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
        resposta = Tria(pergunta,body["nCdUser"])
        return JSONResponse(content={"Pergunta": pergunta, "Resposta":resposta}, status_code=200)
    except Http_Exception as http:
        return JSONResponse(content={"message":http.mensagem}, status_code=http.codigo)
    except Exception as e:
        return JSONResponse(content={"message":e}, status_code=500)


@api.post("/embedding/")
async def embedding():
    try:
        criar_embedding()
        return JSONResponse(content={"message":"Os produtos e ingredientes tiveram o embedding realizado com sucesso"}, status_code=200)
    except Http_Exception as http:
        return JSONResponse(content={"message":http.mensagem}, status_code=http.codigo)
    except Exception as e:
        return JSONResponse(content={"message":e}, status_code=500)
    

@api.post("/scanner/")
async def scanner_tabela(nome_ingrediente:str, file: UploadFile = File(...)):
    try:
        # Lê os bytes do arquivo enviado
        contents = await file.read()

        # Converte os bytes em uma imagem PIL
        image = Image.open(io.BytesIO(contents))
        
        processar_imagem(image, nome_ingrediente)
       
        # Envia a imagem como resposta HTTP
        return JSONResponse(content={"message":f"O ingrediente {nome_ingrediente} foi scanneado e salvo com sucesso"}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"message":e}, status_code=500)
