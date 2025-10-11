from dotenv import load_dotenv
import os
import google.generativeai as genai
from libs.Utils.Exception import Http_Exception
from pathlib import Path
import json

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_GEMINI_API")) 

modelo_tabela = ""
caminho_modelo = Path(__file__).resolve().parent.parent / "docs" / "Models" / "Mongo" / "Tabela.json"


with(open(caminho_modelo, "r", encoding="utf-8")) as arquivo:
    modelo_tabela = arquivo.read()


llm = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    system_instruction=f"""
# Contexto
Você é um especialista em engenharia de alimentos e irá receber um dicionario contendo informações sobre uma tabela nutricional, essas informações devem ser processadas uma por uma e analisadas para gerar uma descrição sobre a qualidade nutricional da tabela.

# Informações sobre a estrutura do dicionário
{modelo_tabela}

# Função
Você deve citar os pontos bons, ruins e recomendações do que deve ser melhorado da tabela nutricional, comparando com outros alimentos.

# Formato de saída
✅ Pontos bons
- <Pontos fortes e comparação com outros alimentos saudáveis>

⚠️ Pontos ruins
- <Pontos fortes e comparação com outros alimentos não saudáveis>

📈 Sugestão de melhoria
- <Sugestões de melhoria>


# Regras
- Seja direto, empático e responsável;
- Evite jargões.
- Mantenha respostas curtas e utilizáveis.

""",
    generation_config=genai.types.GenerationConfig(
        temperature=0.7,
        top_p=0.95,
        # max_output_tokens=5,
        # stop_sequences=["\n\n"]
    )
)


def descrever_avaliacao(tabela:dict):
    try:
        response = llm.generate_content(json.dumps(tabela))
        return response.text
        
    except Exception as e:
        raise Http_Exception(400, f"Erro ao consumir a API para avaliar a tabela nutricional. Erro: {e}")