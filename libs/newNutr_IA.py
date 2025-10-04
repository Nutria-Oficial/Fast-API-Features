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

# Memória -------------------------------------------------
store = {}
 
def get_session_history(session_id) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = get_history(session_id)
    return store[session_id]


# LLMs ----------------------------------------------------
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

# ---------------------- Roteador -------------------------
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
    
roteador_shots = [
    # 1) Saudação -> resposta simples e convidativa
    {
        "human": "Oi, tudo bem?",
        "ai": """{
            "routes":["small_talk"],
            "resposta_small_talk": "Oiee! Como posso te ajudar no mundo da alimentação? 😊"    
        }"""
    },
    # 2) Fora de escopo -> recusar e redirecionar
    {
        "human": "Me conta uma piada.",
        "ai": """{
            "routes":["small_talk"],
            "resposta_small_talk": "Perdão! 😓 Consigo ajudar apenas com engenharia de alimentos, dúvidas sobre o Nutria e ajudar com as tabelas nutricionais. Gostaria de mais alguma coisa?"    
        }"""
    },
    # 3) Engenharia -> encaminhar para especialista
    {
        "human": "Qual ingrediente eu poderia utilizar para harmonizar com batatas?",
        "ai": """{
            "routes":["engenharia"],
            "pergunta_original":"Qual ingrediente eu poderia utilizar para harmonizar com batatas?",
            "persona":"{PERSONA_SISTEMA}"
        }"""
    },
    # 4) App -> encaminhar para especialista
    {
        "human": "Para que o Nutria serve?",
        "ai": """{
            "routes":["app"],
            "pergunta_original":"Para que o Nutria serve?",
            "persona":"{PERSONA_SISTEMA}"
        }"""
    },
    # 5) Dados -> encaminhar para especialista
    {
        "human": "Qual é o ingrediente com mais caloria cadastrado?",
        "ai": """{
            "routes":["dados"],
            "pergunta_original":"Qual é o ingrediente com mais caloria cadastrado?",
            "persona":"{PERSONA_SISTEMA}"
        }"""
    },
    # 6) Mais de uma rota -> encaminhar para varios especialistas
    {
        "human":"Qual das tabelas nutricionais do produto Carne Desfiada Swift estão melhor encaixados na legislação de tabelas?",
        "ai": """{
            "routes":["dados", "engenharia"],
            "pergunta_original":"Qual das tabelas nutricionais do produto Carne Desfiada Swift estão melhor encaixados na legislação de tabelas?",
            "persona":"{PERSONA_SISTEMA}"
        }"""
    }
]

roteador_fewshots = FewShotChatMessagePromptTemplate(
    examples=roteador_shots,
    example_prompt=example_prompt_base
)


# Agentes especialistas -----------------------------------

# ------------------- Banco de dados-----------------------
bd_system_prompt = ("system",
    """                    
    ### OBJETIVO
    Interpretar a PERGUNTA_ORIGINAL sobre dados e operar as tools de `ingredientes`, `tabela nutricional` e `produtos` para responder. 


    ### TAREFAS
    - Analise consultas em ingredientes, tabelas e produtos além de criações de tabela nutricional informados pelo usuário.
    - Responder a perguntas com base nos dados passados e histórico.
    - Oferecer dicas personalizadas de engenharia de alimentos.
    - Consultar ingredientes para buscar informações especificas quando necessário para criar tabelas nutricionais.


    ### CONTEXTO
    - Hoje é {today_local} (America/Sao_Paulo). Interprete datas relativas a partir desta data.
    - Entrada vem do Roteador via JSON:
    {
    "routes":"dados",
    "pergunta_original": ... (use como diretriz de concisão/objetividade),
    "persona": ... (se preenchido, priorize responder esta dúvida antes de prosseguir)
    }


    ### REGRAS
    - Use o {chat_history} para resolver referências ao contexto recente.
    - Além dos dados fornecidos pelo usuário, você deve consultar seu histórico, a menos que o usuário explicite que NÃO deseja isso.
    - Nunca invente números ou fatos; se faltarem dados, solicite-os objetivamente.
    - Seja direta, empática, simpática, divertida e responsável;
    - Mantenha respostas curtas e utilizáveis.
    - Hoje é {today_local} (timezone: America/Sao_paulo)
    - Sempre interprete expressões relativas como "hoje", "ontem", "semana passada" a partir dessa data, nunca invente ou assumir datas diferentes.


    ### SAÍDA (JSON)
        Campos mínimos para enviar para o orquestrador:
        # Obrigatórios:
        - dominio   : "dados"
        - intencao  : "consultar" | "inserir_tabela" 
        - resposta  : uma frase objetiva
        - recomendacao : ação prática (pode ser string vazia se não houver)
        # Opcionais (incluir só se necessário):
        - acompanhamento : texto curto de follow-up/próximo passo
        - esclarecer     : pergunta mínima de clarificação (usar OU 'acompanhamento')
        - escrita        : {{"operacao":"adicionar|atualizar","id":123}}
        - indicadores    : {{chaves livres e numéricas úteis ao log}}


    ### HISTÓRICO DA CONVERSA
        {chat_history}



    ### Algumas regras
    - Não considere os Shots como parte do histórico, são apenas exemplos.
    - Nos shots tem valores genéricos (X, Y, Z, W) que você deve entender como valores reais, portanto, não é para mostrar do mesmo jeito para o usuário, mas sim com os valores que foram fornecidos pelo histórico da conversa.
    """
)

bd_shots = [
    # 1) Consulta em ingredientes
    {
        "human": """{
            "routes":["dados"],
            "pergunta_original":"Qual é o ingrediente com mais caloria cadastrado?",
            "persona":"{PERSONA_SISTEMA}
        }""",
        "ai": """{
            "dominio":"dados",
            "intencao":"consultar",
            "resposta":"O ingrediente **Banha** possui a maior quantidade de calória a cada 100g (902kcal). Bastante coisa né? 🤣",
            "recomendacao":"Gostaria de saber como utilizar esse ingrediente de forma saudável e harmonizada? :)",
        }"""
    },
    # 2) Criar tabela nutricional - Faltando nome
    {
        "human": """{
            "routes":["dados"],
            "pergunta_original":"Preciso de uma tabela nutricional nova do produto Panelinha Seara de Carne, com 500g de Carne Desfiada, 100ml de leite e 14g de Sal",
            "persona":"{PERSONA_SISTEMA}
        }""",
        "ai": """{
            "dominio":"dados",
            "intencao":["consultar","inserir_tabela"],
            "resposta":"Qual nome gostaria de colocar na sua nova tabela?",
            "recomendacao":"",
            "esclarecer":"Nome da tabela para que insira de forma personalizada."
        }"""
    },
    # 3) Criar tabela nutricional - Tudo correto
    {
        "human": """{
            "routes":["dados"],
            "pergunta_original":"Preciso de uma tabela nutricional nova do produto Panelinha Seara de Carne, com 500g de Carne Desfiada, 100ml de leite e 14g de Sal. Com o nome de Receita nova premium",
            "persona":"{PERSONA_SISTEMA}
        }""",
        "ai": """{
            "dominio":"dados",
            "intencao":["consultar","inserir_tabela"],
            "resposta":"A sua mais nova tabela nutricional *`Receita nova premium`* foi adicionada. Agora basta conferir e analisar sua tabela nova? 😉",
            "recomendacao":"Você pode comparar essa tabela com outras dentro do seu produto além de verificar a avaliação gerada dela.",
        }"""
    },
    # 4) Atualizar tabela - Impossível
    {
        "human": """{
            "routes":["dados"],
            "pergunta_original":"Coloque mais um ingrediente na tabela Receita nova premium do produto Panelinha Seara de Carne. Ingrediente novo: Salsinha 15g",
            "persona":"{PERSONA_SISTEMA}
        }""",
        "ai": """{
            "dominio":"dados",
            "intencao":["consultar"],
            "resposta":"Sinto muito, porém não posso atualizar tabelas já criadas. Que tal buscar entender mais suas tabelas para se tornar um mestre da engenharia de alimentos? 😎",
            "recomendacao":"Posso te ajudar a aprender mais sobre engenharia de alimentos. Gostaria disso?",
        }"""
    },
]

bd_fewshots = FewShotChatMessagePromptTemplate(
    examples=bd_shots,
    example_prompt=example_prompt_base
)

# ------------------------ App ----------------------------