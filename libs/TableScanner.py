# Declaração
"""
Após diversos testes, um modelo de OCR para tabelas nutricionais será elegido, seu código ficará neste arquivo até que um novo modelo surja e seja melhor ou mais autêntico que este.

Modelo Atual: IA 2
Descrição do modelo: Utilizando uma chave do Gemini, a imagem é processada e lida inteiramente pelo Gemini, retornando o seu valor, o modelo é rápido e traz os resultados majoritariamente corretos de acordo com nossos testes. Um modelo que utilize o Tesseract e se aproxime da precisão deste irá ser coroado como melhor que este. Porém de acordo com nossas pesquisas, para criar um modelo que não utilize LLMs e que se aproxime dos resultados, ira ser extremamente mais complexo, requisitando um modelo avançado de Machine Learning.
"""

# Modelo
import os
import google.generativeai as genai
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv
from libs.Utils.Connection import get_coll, COLLS,get_highest_id
# from dotenv import load_dotenv


# Configure sua chave da API
# Configuração básica do Gemini
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

# Inicialize o modelo Gemini Pro Vision
def processar_imagem(imagem, nome_ingrediente):
    model = genai.GenerativeModel("gemini-2.0-flash")

    caminho_modelo = Path(__file__).resolve().parent.parent / "docs" / "Models" / "Mongo" / "Ingrediente.json"

    with(open(caminho_modelo, "r", encoding="utf-8")) as arquivo:
        modelo_ingrediente = arquivo.read()


    # Prompt para extrair e converter a imagem em CSV
    prompt = f"""
    Esta imagem contém uma tabela nutricional. Por favor, leia o conteúdo da imagem,
    identifique os dados estruturados (como nutrientes, quantidades em 100g, unidade de medida).
    E retorne os dados encontrados estruturados no seguinte formato:
    {modelo_ingrediente}

    Retorne apenas o que encontrou sem nenhum tipo de comentário extra

    """

    # Gere a resposta
    genai.configure(api_key=api_key)
    response = model.generate_content([prompt, imagem])
    ingrediente = response.text    

    ingrediente = ingrediente.removeprefix("```json\n").removesuffix("```")

    if (ingrediente.startswith("[")):
        ingrediente = dict(list(ingrediente)[0])
    else:
        ingrediente = dict(ingrediente)

    # Realizando as modificações necessárias e inserindo no banco
    cursor = get_coll(COLLS["ingrediente"])
    
    ingrediente["_id"] = get_highest_id(cursor)
    ingrediente["cNmIngrediente"]= nome_ingrediente

    cursor.insert_one(ingrediente)
