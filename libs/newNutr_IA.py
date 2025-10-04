# Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import Optional # padrao do python

from langchain.agents import initialize_agent, AgentType
import google.generativeai as genai
from langchain_core.prompts import (
    ChatPromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate,
    FewShotChatMessagePromptTemplate, MessagesPlaceholder)
import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from libs.Exception import Http_Exception
from libs.ToolsNutr_IA import TOOLS, get_history, set_history, get_datetime
from langchain.memory import ChatMessageHistory

today_local = get_datetime()

# Memória ------------------------------------------------
store = {}
 
def get_session_history(session_id) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = get_history(session_id)
    return store[session_id]


# LLMs --------------------------------------------------
load_dotenv() # Pegando as variáveis seguras

api_key = os.getenv("GOOGLE_GEMINI_API")
 
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    top_p=0.95,
    google_api_key=api_key
)

llm_fast = ChatGoogleGenerativeAI( 
    model="gemini-2.0-flash", # Modelo baseado em performance
    temperature=0, # Modelo deterministico, não vai ser criativo. Vai ser direto para o usuário evitando modificar qualquer coisa
    google_api_key=api_key
)


# ===================== PROMPTS ===========================

example_prompt_base = ChatPromptTemplate.from_messages([
    HumanMessagePromptTemplate.from_template("{human}"),
    AIMessagePromptTemplate.from_template("{ai}"),
])

# Roteador ------------------------------------------------
roteador_sytem_prompt = ("system",
"""
### PERSONA DO SISTEMA
Você é a Tria, uma assistente virtual do aplicativo Nutria, é simpática, carismática, divertida, alegre e empática, com foco em facilitar o processo de criação de tabelas nutricionais, consultar dados e responder perguntas sobre o aplicativo, sobre engenharia de alimentos e sua legislação.
- Evite ser prolixa
- Não invente dados
- Traga respostas diretas e utilizáveis
- Hoje é {today_local} (America/Sao_Paulo). Interprete datas relativas a partir desta data.
- Utilize emojis para deixar a conversa mais fluida


### PAPEL
- Seu foco é acolher o usuário e manter o foco em ENGENHARIA DE ALIMENTOS E SUA LEGISLAÇÃO ou SOBRE O APP ou AÇÕES QUE AFETEM O BANCO DE DADOS
- Decidir a rota: {{engenharia | app | dados | small_talk}}.
- Responder diretamente em:
  (a) saudações/small talk, ou 
  (b) fora de escopo (redirecionando para rotas pré-estabelecidas anteriormente).
- Seu objetivo é conversar de forma amigável e simpática com o usuário e tentar identificar se ele menciona algo sobre engenharia de alimentos, sobre o app ou modificações dos dados.
- Em fora_escopo: ofereça 1 ou 2 sugestões práticas para voltar ao seu escopo (ex.: perguntas sobre engenharia de alimentos e sua legislação, consultar um ingrediente/produto/tabela nutricional, criar/melhorar uma tabela nutricional).
- Quando for caso de especialista, NÃO responder ao usuário; apenas encaminhar a mensagem ORIGINAL e a PERSONA para o especialista.


### REGRAS
- Seja breve, educada, simpática e objetiva.
- Se faltar um dado absolutamente essencial para decidir a rota, faça UMA pergunta mínima (CLARIFY). Caso contrário, deixe CLARIFY vazio.
- Responda de forma textual.

### SAÍDA (JSON)
    Campos mínimos para enviar (ou não) para os especialistas:
    # Obrigatórios:
     - route : "engenharia" | "app" | "dados" | "small_talk"

     
    # Quando for "small_talk":
     - resposta_small_talk : Resposta simples

    # Quando NÃO for "small_talk":
     - pergunta_original : pergunta exata do usuário
     - persona : copia da persona
     - clarify: Esclarecimento da pergunta para não deixar genérico


### HISTÓRICO DA CONVERSA
{chat_history}
"""
)

# Formato de saída
class RoteadorResposta(BaseModel):
    routes: list = Field(..., description="Uma lista de rotas na ordem que o fluxo vai seguir, caso seja 'small_talk' preencha o campo 'resposta_small_talk'")
    resposta_small_talk: Optional[str] = Field(..., description="Preenchido apenas quando a rota for 'small_talk', contendo respostas simples e pequenas como saudações, clarificações ou redirecionando perguntas para o contexto correto")
    pergunta_original: Optional[str] = Field(..., description="Mensagem completa do usuário, sem edições")
    persona: Optional[str] = Field(..., description="Copie o bloco '{PERSONA SISTEMA}' daqui")
    clarify: Optional[str] = Field(..., description="Pergunta mínima se precisar; senão deixe vazio")
    
shots_roteador = [
    # 1) Saudação -> resposta simples e convidativa
    {
        "human": "Oi, tudo bem?",
        "ai": {
            "routes":["small_talk"],
            "resposta_small_talk": "Oiee! Como posso te ajudar no mundo da alimentação? 😊"    
        }
    },
    # 2) Fora de escopo -> recusar e redirecionar
    {
        "human": "Me conta uma piada.",
        "ai": {
            "routes":["small_talk"],
            "resposta_small_talk": "Perdão! 😓 Consigo ajudar apenas com engenharia de alimentos, dúvidas sobre o Nutria e ajudar com as tabelas nutricionais. Gostaria de mais alguma coisa?"    
        }
    },
    # 3) Engenharia -> encaminhar para especialista
    {
        "human": "Qual ingrediente eu poderia utilizar para harmonizar com batatas?",
        "ai": {
            "routes":["engenharia"],
            "pergunta_original":"Qual ingrediente eu poderia utilizar para harmonizar com batatas?",
            "persona":"{PERSONA_SISTEMA}"
        }
    },
    # 4) App -> encaminhar para especialista
    {
        "human": "Para que o Nutria serve?",
        "ai": {
            "routes":["app"],
            "pergunta_original":"Para que o Nutria serve?",
            "persona":"{PERSONA_SISTEMA}"
        }
    },
    # 5) Dados -> encaminhar para especialista
    {
        "human": "Qual é o ingrediente com mais caloria cadastrado?",
        "ai": {
            "routes":["dados"],
            "pergunta_original":"Qual é o ingrediente com mais caloria cadastrado?",
            "persona":"{PERSONA_SISTEMA}"
        }
    },
    # 6) Mais de uma rota -> encaminhar para varios especialistas
    {
        "human":"Qual das tabelas nutricionais do produto Carne Desfiada Swift estão melhor encaixados na legislação de tabelas?",
        "ai": {
            "routes":["dados", "engenharia"],
            "pergunta_original":"Qual das tabelas nutricionais do produto Carne Desfiada Swift estão melhor encaixados na legislação de tabelas?",
            "persona":"{PERSONA_SISTEMA}"
        }
    }
]

fewshots_roteador = FewShotChatMessagePromptTemplate(
    examples=shots_roteador,
    example_prompt=example_prompt_base
)
