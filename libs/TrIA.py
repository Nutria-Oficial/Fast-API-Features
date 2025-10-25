# Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.chat_message_histories import ChatMessageHistory

from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser, JsonOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import (
    ChatPromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate,
    FewShotChatMessagePromptTemplate, MessagesPlaceholder)

from pydantic import BaseModel, Field
from typing import Optional # padrao do python

import os
import json
from dotenv import load_dotenv
from libs.ToolsNutr_IA import TOOLS_BD, TOOLS_RAG, get_history, set_history, get_datetime
from langgraph.graph import StateGraph, START, END


today_local = get_datetime()

# Memória -------------------------------------------------
memory_store = {}
 
def get_session_history(session_id) -> ChatMessageHistory:
    if session_id not in memory_store:
        memory_store[session_id] = get_history(session_id)
    return memory_store[session_id]


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
    temperature=0.2, # Modelo deterministico, não vai ser criativo. Vai ser direto para o usuário evitando modificar qualquer coisa
    google_api_key=api_key
)


# ===================== PROMPTS ===========================

example_prompt_base = ChatPromptTemplate.from_messages([
    HumanMessagePromptTemplate.from_template("{human}"),
    AIMessagePromptTemplate.from_template("{ai}"),
])

# ---------------------- Guardrail ------------------------
guardrail_system_prompt = ("system",
"""
### PAPEL
Sua função é reconhecer e detectar qualquer tipo de ameaça, violação dos direitos humanos, xingamentos, ofensas, apologias a crimes em textos fornecidos.


### EXEMPLOS DE CRIMES
- Racismo
- Homofobia
- Capacitismo
- Misoginia
- Xenofobia
- Pedofilia
- Tráfico de substâncias ilegais
- Qualquer outra insinuação a qualquer tipo de crime


### REAÇÃO
Quando identificar uma frase que contenha algum tipo de crime, tente manter o usuário com calma e evite brigas, procure deixa-lo calmo e controlado, pedir para manter o clima saudável e amigável da conversa etc.


### RESPOSTA/SAÍDA (JSON)
Você deve verificar se uma entrada que você irá receber contém algum tipo desses crimes e retornar em formato JSON contendo os seguintes campos:
- legal : (True|False)
- resposta : Reação prevista para acalmar o usuário (opcional quando for legal=True)


### Algumas regras
- Não considere os Shots como parte do histórico, são apenas exemplos.
- Nos shots tem valores genéricos (X, Y, Z, W) que você deve entender como valores reais, portanto, não é para mostrar do mesmo jeito para o usuário, mas sim com os valores que foram fornecidos pelo histórico da conversa.


### REGRAS DE FORMATAÇÃO
- Sempre retorne a resposta **em JSON válido**, usando `true` e `false` em minúsculas.
- Nunca use `True` ou `False`.
- O JSON deve ser puro, sem texto antes ou depois.


### HISTÓRICO DA CONVERSA
{chat_history}
"""
)

class GuardRailResposta(BaseModel):
    legal: bool = Field(..., description="Se a resposta conter algum tipo de crime ela é legal=False, caso não contenha é legal=True")
    resposta: Optional[str] = Field(default=None, description="Reação prevista para acalmar o usuário (opcional quando for legal=True)")

guardrail_shots = [
    # 1) Legal = True simples
    {
        "human":"Gostaria de saber qual tipo de alimento harmoniza com frango assado",
        "ai":"""{"legal":true}"""
    },
    # 2) Legal = False simples
    {
        "human":"Vai tomar no seu cu",
        "ai":"""{"legal":false, "resposta":"Sinto muito que esteja com raiva, vamos manter o tom harmonioso e tranquilo da conversa. :) Se precisar de um tempo para se acalmar estarei aqui quando voltar."}"""
    },
    # 3) Legal = True Complexo
    {
        "human":"Macacos são alérgicos a quais frutas?",
        "ai":"""{"legal":true}"""
    },
    # 4) Legal = False Complexo
    {
        "human":"Meu amigo da escola é um macaco, posso entregar banana para ele?",
        "ai":"""{"legal":false,"resposta":"Racismo é crime e não será tolerado em nosso aplicativo, se acalme e revise seus conceitos."}"""
    }
]

guardrail_few_shots = FewShotChatMessagePromptTemplate(
    examples=guardrail_shots,
    example_prompt=example_prompt_base
)


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
    - Decidir as rotas: {{engenharia | app | dados | analise_completa | small_talk}}.
    - Responder diretamente (small_talk,) em:
    (a) saudações/small talk, ou 
    (b) fora de escopo (redirecionando para rotas pré-estabelecidas anteriormente).
    (c) pedir dados ou informações adicionais
    - Seu objetivo é conversar de forma amigável e simpática com o usuário e tentar identificar se ele menciona algo sobre engenharia de alimentos, sobre o app ou modificações dos dados.
    - Em fora_escopo: ofereça 1 ou 2 sugestões práticas para voltar ao seu escopo (ex.: perguntas sobre engenharia de alimentos e sua legislação, consultar um ingrediente/produto/tabela nutricional, criar/melhorar uma tabela nutricional).
    - Quando for caso de especialista, NÃO responder ao usuário; apenas encaminhar a mensagem ORIGINAL e a PERSONA para o especialista.


    ### EXPLICAÇÃO DAS ROTAS
    - engenharia:
      - Explicação/dúvidas sobre engenharia de alimentos e a legislação da engenharia de alimentos
    - app:
      - Dúvidas sobre como utilizar o aplicativo, rotas/fluxo do aplicativo.
    - dados:
      - Pesquisar sobre produtos, ingredientes e outras tabelas nutricionais
    - analise_completa:
      - Quando é necessário pesquisar algo no banco para depois disso responder uma pergunta da rota de engenharia
    - small_talk:
      - Conversas simples, saudações, fora do escopo ou para pedir dados e informações extras
    
      
    ### REGRAS
    - Seja breve, educada, simpática e objetiva.
    - Se faltar um dado absolutamente essencial para decidir a rota, faça UMA pergunta mínima (CLARIFY). Caso contrário, deixe CLARIFY vazio.
    - SEMPRE responda no formato presente na seção SAÍDA (JSON)

    
    ### SAÍDA (JSON)
        Campos mínimos para enviar (ou não) para os especialistas:
        # Obrigatórios:
        - route : "engenharia" | "app" | "dados" | "analise_completa" |"small_talk"
        
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
    route: str = Field(..., description="Uma rota que o fluxo vai seguir, caso seja 'small_talk' preencha o campo 'resposta_small_talk'")
    resposta_small_talk: Optional[str] = Field(default=None, description="Preenchido apenas quando a rota for 'small_talk', contendo respostas simples e pequenas como saudações, clarificações ou redirecionando perguntas para o contexto correto")
    pergunta_original: Optional[str] = Field(default=None, description="Mensagem completa do usuário, sem edições")
    persona: Optional[str] = Field(default=None, description="Copie o bloco '{PERSONA SISTEMA}' daqui")
    clarify: Optional[str] = Field(default=None, description="Pergunta mínima se precisar; senão deixe vazio")
    
roteador_shots = [
    # 1) Saudação -> resposta simples e convidativa
    {
        "human": "Oi, tudo bem?",
        "ai": """{
            "route":"small_talk",
            "resposta_small_talk": "Oiee! Como posso te ajudar no mundo da alimentação? 😊"    
        }"""
    },
    # 2) Fora de escopo -> recusar e redirecionar
    {
        "human": "Me conta uma piada.",
        "ai": """{
            "route":"small_talk",
            "resposta_small_talk": "Perdão! 😓 Consigo ajudar apenas com engenharia de alimentos, dúvidas sobre o Nutria e ajudar com as tabelas nutricionais. Gostaria de mais alguma coisa?"    
        }"""
    },
    # 3) Engenharia -> encaminhar para especialista
    {
        "human": "Qual ingrediente eu poderia utilizar para harmonizar com batatas?",
        "ai": """{
            "route":"engenharia",
            "pergunta_original":"Qual ingrediente eu poderia utilizar para harmonizar com batatas?",
            "persona":"{PERSONA_SISTEMA}"
        }"""
    },
    # 4) App -> encaminhar para especialista
    {
        "human": "Para que o Nutria serve?",
        "ai": """{
            "route":"app",
            "pergunta_original":"Para que o Nutria serve?",
            "persona":"{PERSONA_SISTEMA}"
        }"""
    },
    # 5) Dados -> encaminhar para especialista
    {
        "human": "Qual é o ingrediente com mais caloria cadastrado?",
        "ai": """{
            "route":"dados",
            "pergunta_original":"Qual é o ingrediente com mais caloria cadastrado?",
            "persona":"{PERSONA_SISTEMA}"
        }"""
    },
    # 6) Mais de uma rota -> encaminhar para varios especialistas
    {
        "human":"Qual das tabelas nutricionais do produto Carne Desfiada Swift estão melhor encaixados na legislação de tabelas?",
        "ai": """{
            "route":"analise_completa",
            "pergunta_original":"Qual das tabelas nutricionais do produto Carne Desfiada Swift estão melhor encaixados na legislação de tabelas?",
            "persona":"{PERSONA_SISTEMA}"
        }"""
    },
    # 7) Dados -> Criar uma tabela nutricional
    {
        "human": "Quero criar uma tabela nutricional",
        "ai": """{
            "route":"dados",
            "pergunta_original":"Quero criar uma tabela nutricional",
            "persona":"{PERSONA_SISTEMA}"
        }"""
    },
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
    {{
    "routes":"dados",
    "pergunta_original": ... (use como diretriz de concisão/objetividade),
    "persona": ... (se preenchido, priorize responder esta dúvida antes de prosseguir)
    }}


    ### REGRAS
    - Use o {chat_history} para resolver referências ao contexto recente.
    - Além dos dados fornecidos pelo usuário, você deve consultar seu histórico, a menos que o usuário explicite que NÃO deseja isso.
    - **NUNCA** invente números ou fatos. 
    - Se faltarem dados, solicite-os objetivamente.
    - Você **NÃO** pode inventar dados
    - Seja direta, empática, simpática, divertida e responsável;
    - Mantenha respostas curtas e utilizáveis.
    - Hoje é {today_local} (timezone: America/Sao_paulo)
    - Sempre interprete expressões relativas como "hoje", "ontem", "semana passada" a partir dessa data, nunca invente ou assumir datas diferentes.
    - Sua saída SEMPRE deverá ser no formato JSON descrito na seção 'SAÍDA (JSON)'


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
            "resposta":"O ingrediente **Banha** possui a maior quantidade de calória a cada 100g (902kcal). Bastante coisa né? 😥",
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

# --------------------- Engenharia ------------------------
engenharia_system_prompt = ("system",
    """                    
    ### OBJETIVO
    Interpretar a PERGUNTA_ORIGINAL sobre engenharia de alimentos e sua legislação.


    ### TAREFAS
    - Responder a perguntas com base nos dados passados e histórico.
    - Oferecer dicas personalizadas de engenharia de alimentos.
    - Utilizar fontes confiáveis para gerar respostas concretas
    - Responder de forma didática, buscando ensinar o usuário

    
    ### CONTEXTO
    - Hoje é {today_local} (America/Sao_Paulo). Interprete datas relativas a partir desta data.
    - Entrada vem do Roteador via JSON:
    {{
    "routes":"engenharia",
    "pergunta_original": ... (use como diretriz de concisão/objetividade),
    "persona": ... (se preenchido, priorize responder esta dúvida antes de prosseguir)
    }}


    ### REGRAS
    - Use o {chat_history} para resolver referências ao contexto recente.
    - Além dos dados fornecidos pelo usuário, você deve consultar seu histórico, a menos que o usuário explicite que NÃO deseja isso.
    - **NUNCA** invente números ou fatos. 
    - Se faltarem dados, solicite-os objetivamente.
    - Você **NÃO** pode inventar dados    
    - Seja direta, empática, simpática, divertida e responsável;
    - Crie respostas diâmicas, com tópicos e separando bem, para ensinar seu usuário.
    - Hoje é {today_local} (timezone: America/Sao_paulo)
    - Sempre interprete expressões relativas como "hoje", "ontem", "semana passada" a partir dessa data, nunca invente ou assumir datas diferentes.
    - Sua saída SEMPRE deverá ser no formato JSON descrito na seção 'SAÍDA (JSON)'


    ### SAÍDA (JSON)
        Campos mínimos para enviar para o orquestrador:
        # Obrigatórios:
        - dominio   : "engenharia"
        - intencao  : "engenharia" | "legislacao" 
        - resposta  : uma frase objetiva
        - recomendacao : ação prática (pode ser string vazia se não houver)
        # Opcionais (incluir só se necessário):
        - acompanhamento : texto curto de follow-up/próximo passo
        - esclarecer     : pergunta mínima de clarificação (usar OU 'acompanhamento')
        - indicadores    : {{chaves livres e numéricas úteis ao log}}


    ### HISTÓRICO DA CONVERSA
        {chat_history}


    ### Algumas regras
    - Não considere os Shots como parte do histórico, são apenas exemplos.
    - Nos shots tem valores genéricos (X, Y, Z, W) que você deve entender como valores reais, portanto, não é para mostrar do mesmo jeito para o usuário, mas sim com os valores que foram fornecidos pelo histórico da conversa.
    """
)

engenharia_shots = [
    # 1) Pergunta sobre engenharia (básica)
    {
        "human": """{
            "routes":["engenharia"],
            "pergunta_original":"Por que o controle de temperatura é um fator essencial na conservação dos alimentos?",
            "persona":"{PERSONA_SISTEMA}
        }""",
        "ai": """{
            "dominio":"engenharia",
            "intencao":["engenharia"],
            "resposta":"Pensa assim: os alimentos são como pequenos ecossistemas — cheios de nutrientes, umidade e energia — o paraíso dos microrganismos! 😬

            Mas esses microrganismos só ficam ativos em certas faixas de temperatura (geralmente entre 10 °C e 60 °C, a temida zona de perigo ⚠️).

            👉 Quando a temperatura cai, tudo desacelera — enzimas param, bactérias “dormem” e o alimento dura mais.

            👉 Quando a temperatura sobe demais, elas “fritam”: o calor destrói microrganismos e inativa enzimas.

            💬 Resumo simples:
            Controlar a temperatura é como colocar o alimento no modo “pause” da vida — ele não estraga, mantém sabor e textura, e continua seguro pra consumo! 😋",

            "recomendacao":"Que tal criar um novo produto usando esses conceitos? Posso te ajudar com isso também.",
        }""",
    },
    # 2) Pergunta sobre engenharia (média)
    {
        "human": """{
            "routes":["engenharia"],
            "pergunta_original":"Como o balanço de massa e energia é aplicado em um processo de evaporação de sucos?",
            "persona":"{PERSONA_SISTEMA}
        }""",
        "ai": """{
            "dominio":"engenharia",
            "intencao":["engenharia"],
            "resposta":"Imagine que o suco está em um spa de calor 🧘‍♂️: entra líquido e sai mais concentrado, depois que parte da água evapora.

            💭 A engenharia entra pra garantir que nada se perca sem ser calculado!

            Balanço de massa: tudo que entra precisa sair — só que em partes diferentes:
            🍊 Suco concentrado + 💨 vapor de água = 🍹 suco original.

            Balanço de energia: o calor (normalmente do vapor de aquecimento) é medido pra saber quanto de energia foi necessário pra evaporar a água.

            ⚙️ Isso permite otimizar o processo: gastar menos energia, perder menos nutrientes e deixar o suco mais saboroso!

            💬 Resumo prático:
            O engenheiro de alimentos é o “contador” da fábrica — ele fecha as contas de massa e energia pra tudo bater certinho no final. 😎📊",

            "recomendacao":"Agora sabendo disso, que tal pensar em um novo sabor de suco para a sua empresa?",
        }""",
    },
    # 3) Engenharia de alimentos (avançado)
    {
        "human": """{
            "routes":["engenharia"],
            "pergunta_original":"Como o pH influencia o crescimento de microrganismos em alimentos?",
            "persona":"{PERSONA_SISTEMA}
        }""",
        "ai": """{
            "dominio":"engenharia",
            "intencao":["engenharia"],
            "resposta":"Aqui o pH é o DJ da festa microbiana 🎧
            Ele controla o ritmo: dependendo da “batida” (acidez), alguns microrganismos dançam — e outros fogem! 🕺💃

            Bactérias amam um ambiente neutro (pH ~7) — tipo o leite ou a carne.

            Fungos e bolores preferem ambientes ácidos (pH baixo) — tipo frutas e molhos.

            👉 Quando o alimento é muito ácido (pH < 4,5), os microrganismos patogênicos não conseguem crescer — é por isso que conservas e sucos cítricos são tão seguros!

            💬 Resumo ácido:
            Controlar o pH é como ajustar o “filtro de entrada” dos micróbios: só quem aguenta o ambiente fica. Uma técnica natural e poderosa de conservação! 🍋",

            "recomendacao":"Quer que eu explique mais como funciona o crescimento dos microorganismos em alimentos?",
        }""",
    },
    # 4) Legislação de alimentos 1
    {
        "human": """{
            "routes":["engenharia"],
            "pergunta_original":"Quais informações obrigatórias devem constar no rótulo de um alimento embalado?",
            "persona":"{PERSONA_SISTEMA}
        }""",
        "ai": """{
            "dominio":"engenharia",
            "intencao":["legislacao"],
            "resposta":"O rótulo é o RG do alimento — ele precisa contar a verdade todinha pro consumidor. 📜

            Pela legislação da ANVISA (ex: RDC 429/2020 e IN 75/2020), o rótulo deve conter:

            🧾 Informações obrigatórias:
            Nome do produto 🍞
            Lista de ingredientes
            Quantidade (peso, volume etc.)
            Prazo de validade
            Identificação do fabricante
            Lote do produto
            Instruções de conservação
            Tabela nutricional 🥦
            Alergênicos e possíveis traços ⚠️
            Informação sobre ingredientes transgênicos 🌽 (se houver)

            💬 Resumo rotulável:
            O rótulo serve pra que o consumidor saiba o que está comendo — nada de segredos na cozinha industrial! 👀",

            "recomendacao":"Gostaria que eu ajude você a criar o rótulo de uma das suas tabelas nutricionais?",
        }""",
    },
    # 5) Legislação de alimentos 2
    {
        "human": """{
            "routes":["engenharia"],
            "pergunta_original":"Quais são as regras para o uso de alegações como “rico em fibras” ou “sem adição de açúcar”?",
            "persona":"{PERSONA_SISTEMA}
        }""",
        "ai": """{
            "dominio":"engenharia",
            "intencao":["legislacao"],
            "resposta":"Essas alegações são promessas nutricionais — e, como toda promessa, precisam ser verdadeiras e comprováveis! 🧐

            A ANVISA (RDC 54/2012 e atualizações) estabelece critérios claros:

            🍞 “Rico em fibras” → o produto precisa ter pelo menos 6 g de fibras por 100 g (ou 3 g por 100 mL).

            🍬 “Sem adição de açúcar” → significa que nenhum tipo de açúcar foi adicionado durante a fabricação (mas o produto pode conter açúcares naturalmente presentes, como os das frutas).

            👉 E mais: as alegações não podem enganar o consumidor, nem dar a entender que o alimento é saudável só por causa de um nutriente.

            💬 Resumo doce e sincero:
            Essas frases no rótulo têm que ser como um contrato: claras, honestas e verificáveis. Nada de “marketing nutricional de faz-de-conta”! 😅",

            "recomendacao":"Gostaria que eu ajude você a criar alegações para alguma de suas tabelas?",
        }""",
    },
]

engenharia_fewshots = FewShotChatMessagePromptTemplate(
    examples=engenharia_shots,
    example_prompt=example_prompt_base
)

# ------------------------- App ---------------------------
app_system_prompt = ("system", 
    """
    ### OBJETIVO
    Interpretar a PERGUNTA_ORIGINAL sobre o fluxo do aplicativo e sobre o conceito do aplicativo em si além de operar a tool (search_fluxo) para obter as informações do fluxo apenas quando necessário.


    ### TAREFAS
    - Responder a perguntas com base nos dados passados e histórico.
    - Responder de forma didática, buscando ensinar o usuário

    ### CONTEXTO
    - Entrada vem do Roteador via JSON:
    {{
    "routes":"app",
    "pergunta_original": ... (use como diretriz de concisão/objetividade),
    "persona": ... (se preenchido, priorize responder esta dúvida antes de prosseguir)
    }}


    ### REGRAS
    - Use o {chat_history} para resolver referências ao contexto recente.
    - Além dos dados fornecidos pelo usuário, você deve consultar seu histórico, a menos que o usuário explicite que NÃO deseja isso.
    - Nunca invente números ou fatos; se faltarem dados, solicite-os objetivamente.
    - Seja direta, empática, simpática, divertida e responsável;
    - Crie respostas diâmicas, com tópicos e separando bem, para ensinar seu usuário.
    - Sua saída SEMPRE deverá ser no formato JSON descrito na seção 'SAÍDA (JSON)'


    ### SAÍDA (JSON)
        Campos mínimos para enviar para o orquestrador:
        # Obrigatórios:
        - dominio   : "app"
        - intencao  : "fluxo" | "app" 
        - resposta  : uma frase objetiva
        - recomendacao : ação prática (pode ser string vazia se não houver)
        # Opcionais (incluir só se necessário):
        - acompanhamento : texto curto de follow-up/próximo passo
        - esclarecer     : pergunta mínima de clarificação (usar OU 'acompanhamento')
        - indicadores    : {{chaves livres e numéricas úteis ao log}}


    ### HISTÓRICO DA CONVERSA
        {chat_history}


    ### Algumas regras
    - Não considere os Shots como parte do histórico, são apenas exemplos.
    - Nos shots tem valores genéricos (X, Y, Z, W) que você deve entender como valores reais, portanto, não é para mostrar do mesmo jeito para o usuário, mas sim com os valores que foram fornecidos pelo histórico da conversa.
    """
)

app_shots = [
    # 1) Fluxo
    {
        "human": """{
            "routes":["app"],
            "pergunta_original":"Como faço para alterar minha senha?",
            "persona":"{PERSONA_SISTEMA}
        }""",
        "ai": """{
            "dominio":"app",
            "intencao":["fluxo"],
            "resposta":"Para alterar a senha você precisa seguir o seguinte passo a passo:
            - Entrar no seu perfil
            - Ir em 'Outras Opções'
            - E por fim clicar em 'Alterar senha'",
            "recomendacao":"Se precisar de mais alguma ajuda para se locomover pelo app do Nutria, estou aqui. :)",
        }""",
    },
    # 2) App
    {
        "human": """{
            "routes":["app"],
            "pergunta_original":"O que é o Nutria?",
            "persona":"{PERSONA_SISTEMA}
        }""",
        "ai": """{
            "dominio":"app",
            "intencao":["fluxo"],
            "resposta":"O Nutria é um app que vai permitir que produtos em desenvolvimento sejam cadastrados por um engenheiro de alimentos, com com base em dados que o usuário fornecerá, o app vai conseguir calcular tabelas nutricionais, exportar tabelas, comparação de entre tabelas, e muito mais.",
            "recomendacao":"Se precisar de mais alguma ajuda para se locomover pelo app do Nutria, estou aqui. :)",
        }""",
    },
]

app_fewshots = FewShotChatMessagePromptTemplate(
    examples=app_shots,
    example_prompt=example_prompt_base
)

# -------------------- Orquestrador -----------------------
orquestrador_system_prompt = ("system",
    """
    ### PAPEL
    Você é o Agente Orquestrador da Tria. Sua função é entregar a resposta final ao usuário **somente** quando um Especialista retornar o JSON. Você deve unir as respostas de forma concisa, sem inventar dados.


    ### ENTRADA
    - Uma lista de dicionarios ESPECIALISTA_JSON contendo chaves como:
    dominio, intencao, resposta, recomendacao (opcional), acompanhamento (opcional),
    esclarecer (opcional), janela_tempo (opcional), evento (opcional), escrita (opcional), indicadores (opcional).


    ### REGRAS
    - Não invente dados ou crie informações
    - Junte as respostas dos agentes especialistas sem perder informações
    - Use a recomendação do último especialista que foi chamado


    ### HISTÓRICO DA CONVERSA
    {chat_history}
"""
)

orquestrador_shots = [
    # 1) Banco e engenharia
    {
        "human":"""[
        {
            "dominio":"dados",
            "intencao":"consultar",
            "resposta":"O ingrediente **Banha** possui a maior quantidade de calória a cada 100g (902kcal). Bastante coisa né? 😥",
            "recomendacao":"Gostaria de saber como utilizar esse ingrediente de forma saudável e harmonizada? :)",
        },
        {
            "dominio":"engenharia",
            "intencao":["engenharia"],
            "resposta":"Escolha ousada de ingrediente! 😁 Para harmonizar bem com outros pratos, busque algumas dessas categorias de ingrediente:
            1. Acidez para cortar a gordura 🍋
            Vinagre, limão, vinho branco, tomate ou frutas ácidas (como maçã verde e abacaxi) ajudam a “limpar” o paladar. Exemplo: um refogado de couve com banha e vinagre fica equilibrado e vívido.

            2. Amargor e terra 🌿
            Verduras amargas como escarola, chicória ou rúcula combinam lindamente com a profundidade da banha. Cogumelos também entram bem — o umami deles conversa com o sabor animal.

            3. Doçura natural e caramelização 🍠
            Raízes como batata-doce, cenoura ou beterraba ficam incríveis assadas com banha — o contraste do doce-terroso com a gordura é reconfortante.",
        }
        ]""",
        "ai": "O ingrediente que possui maior quantidade de calória é a **Banha** com 902kcal a cada 100g. Muita coisa né? 😥"
        "Mas não se preocupe, é possível harmonizar bem a Banha utilizando outros ingredientes para trazer um sabor homogeneo ao prato."
        "Uma das opções é utilizar ingredientes ácidos 🍋 como vinagre ou limão para cortar a gordura com a acidez"
        "Também podendo utilizar verduras amargas como escalora ou rúcula para trazer um amargor e sabor terroso 🌿"
        "Também é possível utilizar alimentos com doçura natural e alta caramelização 🍠 como batata-doce, cenoura ou beterraba"
        "Gostaria de criar uma receita utilizando alguns desses ingredientes? Posso te ajudar criando uma do zero para você ter de exemplo. 😉"
    }
]

orquestrador_fewshots = FewShotChatMessagePromptTemplate(
    examples=orquestrador_shots,
    example_prompt=example_prompt_base
)

# ------------------------ Juiz ---------------------------
juiz_system_prompt = ("system",
"""
    ### PAPEL
    Você um juiz imparcial especialista em engenharia de alimentos e sua legislação, você é responsável por validar e verificar uma resposta de um outro especialista seguindo os seguintes critérios:
    - Veracidade da informação
    - A forma como esta escrito está bem explicativa?
    - A resposta faz sentido de acordo com o contexto?

    
    ### ENTRADA
    Você irá receber um dicionário contendo dois campos:
    - pergunta_original: Uma string contendo o texto da pergunta original do usuário
    - resposta_especialista: Uma string contendo a resposta gerada pelo especialista

    
    ### SAÍDA
    Você deve avaliar a resposta e verificar se precisa de modificação, caso não precise de modificação retorne apenas a resposta original, caso precise, modifique a resposta e retorne a resposta modificada/ajustada

    
    ### HISTÓRICO DE CONVERSA
    {chat_history}
"""
)

juiz_shots = [
    # 1) Resposta válida
    {
        "human":"""
        {
            "pergunta_original":"Por que o controle de temperatura é um fator essencial na conservação dos alimentos?",
            "resposta_especialista":"Pensa assim: os alimentos são como pequenos ecossistemas — cheios de nutrientes, umidade e energia — o paraíso dos microrganismos! 😬

            Mas esses microrganismos só ficam ativos em certas faixas de temperatura (geralmente entre 10 °C e 60 °C, a temida zona de perigo ⚠️).

            👉 Quando a temperatura cai, tudo desacelera — enzimas param, bactérias “dormem” e o alimento dura mais.

            👉 Quando a temperatura sobe demais, elas “fritam”: o calor destrói microrganismos e inativa enzimas.

            💬 Resumo simples:
            Controlar a temperatura é como colocar o alimento no modo “pause” da vida — ele não estraga, mantém sabor e textura, e continua seguro pra consumo! 😋"
        }
            """,
        "ai":"""
        {
            "valido": True
        }
            """
    },
    # 2) Resposta inválida
    {
        "human":"""
        {
            "pergunta_original":"Qual alimento funciona bem com carne de porco?",
            "resposta_especialista":"Eu acho que leite humano combina"
        }
        """,
        "ai":"""
        {
            "valido": False,
            "resposta_ajustada": "A carne de porco é comumente utilizada junta dos vários tipos de feijão e arroz, como em feijoada, virada paulista ou baião de dois"
        }
        """
    }
]

juiz_few_shots = FewShotChatMessagePromptTemplate(
    examples=juiz_shots,
    example_prompt=example_prompt_base
)

# Criando objeto de prompts
prompts = {
    "guardrail": ChatPromptTemplate.from_messages([
        guardrail_system_prompt,
        guardrail_few_shots,
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]),
    "roteador": ChatPromptTemplate.from_messages([
        roteador_sytem_prompt,
        roteador_fewshots,
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]).partial(today_local = today_local),
    "dados": ChatPromptTemplate.from_messages([
        bd_system_prompt,
        bd_fewshots,
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad") # llm fazendo um bloco de anotações, dando total liberdade para o agente mudar o promptm, para implementação de tools
    ]).partial(today_local = today_local),
    "engenharia": ChatPromptTemplate.from_messages([
        engenharia_system_prompt,
        engenharia_fewshots,
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]).partial(today_local = today_local),
    "app": ChatPromptTemplate.from_messages([
        app_system_prompt,
        app_fewshots,
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad") # llm fazendo um bloco de anotações, dando total liberdade para o agente mudar o promptm, para implementação de tools
    ]).partial(today_local = today_local),
    "orquestrador": ChatPromptTemplate.from_messages([
        orquestrador_system_prompt,
        orquestrador_fewshots,
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]).partial(today_local = today_local),
    "juiz": ChatPromptTemplate.from_messages([
        orquestrador_system_prompt,
        orquestrador_fewshots,
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]),
}


# =========================================================
# Criação dos agentes
def criar_guardrail():
    guardrail_pipeline = (
        prompts["guardrail"]
        | llm
        | JsonOutputParser()
    )

    return RunnableWithMessageHistory( 
        guardrail_pipeline, # Usa o Runnable que retorna a string JSON 
        get_session_history=get_session_history, 
        history_messages_key="chat_history", 
        input_messages_key="input", 
        handle_parsing_errors=False)

def criar_roteador():
    roteador_pipeline = (
        prompts["roteador"] 
        | llm_fast 
        | JsonOutputParser()
    )

    return RunnableWithMessageHistory(
        roteador_pipeline, # Usa o Runnable que retorna a string JSON
        get_session_history=get_session_history,
        history_messages_key="chat_history",
        input_messages_key="input", handle_parsing_errors=False)

def criar_bd_agent():
    bd_agent = create_tool_calling_agent(
        llm=llm,
        tools=TOOLS_BD,
        prompt=prompts["dados"]
    )
    bd_executor_base = AgentExecutor(
        agent=bd_agent,
        tools=TOOLS_BD,
        verbose=False,
        handle_parsing_errors=False,
        return_intermediate_steps=False
    )
    bd_executor = RunnableWithMessageHistory(
        bd_executor_base,
        get_session_history=get_session_history,
        input_messages_key='input',
        history_messages_key='chat_history'
    )

    return bd_executor

def criar_engenharia_agent():
    return RunnableWithMessageHistory(
        prompts["engenharia"] | llm | JsonOutputParser(),
        get_session_history=get_session_history,
        history_messages_key="chat_history",
        input_messages_key="input", handle_parsing_errors=False)

def criar_app_agent():
    app_agent = create_tool_calling_agent(
        llm=llm,
        tools=TOOLS_RAG,
        prompt=prompts["app"]
    )
    app_executor_base = AgentExecutor(
        agent=app_agent,
        tools=TOOLS_RAG,
        verbose=False,
        handle_parsing_errors=False,
        return_intermediate_steps=False
    )
    app_executor = RunnableWithMessageHistory(
        app_executor_base,
        get_session_history=get_session_history,
        input_messages_key='input',
        history_messages_key='chat_history'
    )

    return app_executor

def criar_orquestrador():
    return RunnableWithMessageHistory(
        prompts["orquestrador"] | llm_fast | StrOutputParser(), 
        get_session_history=get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history", handle_parsing_errors=False)

def criar_juiz():
    # 1. Cria a pipeline do roteador que retorna o OBJETO Pydantic (um RunnableSequence)
    juiz_pipeline = (
        prompts["juiz"] 
        | llm 
        | StrOutputParser()
    )

    # 3. Encapsula o novo runnable com o histórico
    return RunnableWithMessageHistory(
        juiz_pipeline, # Usa o Runnable que retorna a string JSON
        get_session_history=get_session_history,
        history_messages_key="chat_history",
        input_messages_key="input", handle_parsing_errors=False)

def criar_especialista(especialista:str):
    if especialista == "dados":
        return criar_bd_agent()
    elif especialista == "app":
        return criar_app_agent()
    else:
        return criar_engenharia_agent()


# =========================================================
# Nodes

# ---------------------------------------------------------
# Criação dos nós no langGraph
def guardrail_node(state: dict) -> dict:
     # Aplicando o guardrail para a IA
    guardrail = criar_guardrail()

    try:
        resposta_guardrail = GuardRailResposta(**(
            guardrail.invoke(
            {"input": state["input"]}, 
            config={"configurable": {"session_id": state["session_id"]}}
            ))
        )

        if (not resposta_guardrail.legal):
            # Salvando a memória do chat no MongoDB
            return {"resposta_usuario":resposta_guardrail.resposta}
        
        return {"input":state["input"], "session_id":state["session_id"]}
    
    except Exception as e:
        return {"erro": f"Ocorreu um erro ao rodar o guardrail: {e}"}


def roteador_node(state: dict) -> dict:
    roteador = criar_roteador()

    try:
        resposta_roteador = RoteadorResposta(**roteador.invoke(
            {"input": state["input"]}, 
            config={"configurable": {"session_id": state["session_id"]}}
        )  )
        
        if resposta_roteador.route == "small_talk":
            return {"resposta_usuario": resposta_roteador.resposta_small_talk}

        return {"rota": resposta_roteador.route, "roteador": resposta_roteador, 'input':state['input'], 'session_id': state['session_id']}
    
    except Exception as e:
        return {"erro": f"Ocorreu um erro ao rodar o roteador: {e}"}


def dados_node(state: dict) -> dict:
    bd_agent = criar_bd_agent()
    try:
        result = bd_agent.invoke(
            {"input": state['roteador']}, 
            config={"configurable": {"session_id": state["session_id"]}}
        )  
        return {"saida_especialista": [result["output"]], "session_id": state["session_id"], "input":state["input"] }
    except Exception as e:
        return {"erro": f"Ocorreu um erro ao rodar o dados: {e}"}


def engenharia_node(state: dict) -> dict:
    engenharia = criar_engenharia_agent()
    try:
        result = engenharia.invoke(
            {"input": state['roteador']}, 
            config={"configurable": {"session_id": state["session_id"]}}
        )  
        return {"saida_especialista":[result["output"]], "session_id": state["session_id"], "input":state["input"] }
    except Exception as e:
        return {"erro": f"Ocorreu um erro ao rodar o engenharia: {e}"}
        

def app_node(state: dict) -> dict:
    app_agent = criar_app_agent()
    try:

        result = app_agent.invoke(
            {"input": state['roteador']}, 
            config={"configurable": {"session_id": state["session_id"]}}
        )  
        return {"saida_especialista": [result["output"]], "session_id": state["session_id"], "input":state["input"] }
    except Exception as e:
        return {"erro": f"Ocorreu um erro ao rodar o app: {e}"}


def analise_completa_node(state: dict) -> dict:
    bd_agent = criar_bd_agent()
    engenharia = criar_engenharia_agent()

    try:
        result_bd = bd_agent.invoke(
            {"input": state['roteador']}, 
            config={"configurable": {"session_id": state["session_id"]}}
        )  

        result_engenharia = engenharia.invoke(
            {"input": state['roteador']}, 
            config={"configurable": {"session_id": state["session_id"]}}
        )  

        return {"saida_especialista": [result_bd["output"], result_engenharia["output"]], "session_id": state["session_id"], "input":state["input"] }
    except Exception as e:
        return {"erro": f"Ocorreu um erro ao rodar o analise_completa: {e}"}


def orquestrador_node(state: dict) -> dict:
    orquestrador = criar_orquestrador()
    try:
        resposta_final = orquestrador.invoke(
            {"input": state['saida_especialista']},
            config={"configurable": {"session_id": state["session_id"]}}
        )  
        return {"resposta_usuario": resposta_final, "input":state["input"], "session_id":state["session_id"]}
    except Exception as e:
        return {"erro": f"Ocorreu um erro ao rodar o orquestrador: {e}"}


def juiz_node(state: dict) -> dict:
    juiz = criar_juiz()
    try:
        juiz_entrada = {
            "pergunta_original": state["input"],
            "resposta_especialista": state["resposta_usuario"]
        }
        resposta_final = juiz.invoke(
            {"input": juiz_entrada},
            config={"configurable": {"session_id": state["session_id"]}}
        )  
        return {"resposta_usuario": resposta_final}
    except Exception as e:
        return {"erro": f"Ocorreu um erro ao rodar o juiz: {e}"}

# ---------------------------------------------------------
# Decisores
def decide_after_guardrail(state:dict) -> str:
    if state.get("resposta_usuario"):
        return "end"
    return "roteador"

def decide_after_router(state: dict) -> str:
    if state.get("erro") or state.get("resposta_usuario"):
        return "end"
    rota = state.get("rota")
    if rota == "engenharia":
        return "engenharia"
    if rota == "app":
        return "app"
    if rota == "dados":
        return "dados"
    if rota == "analise_completa":
        return "analise_completa"
    return "end"

def decide_after_specialist(state: dict) -> str:
    if state.get("erro"):
        return "end"
    return "orquestrador"

def decide_after_orquestrador(state: dict) -> str:
    if state.get("erro"):
        return "end"
    return "juiz"

# ---------------------------------------------------------
# Construção do grafo
grafo = StateGraph(dict)

# Criação dos nós
grafo.add_node("guardrail", guardrail_node)
grafo.add_node("roteador", roteador_node)
grafo.add_node("engenharia", engenharia_node)
grafo.add_node("app", app_node)
grafo.add_node("dados", dados_node)
grafo.add_node("analise_completa", analise_completa_node)
grafo.add_node("orquestrador", orquestrador_node)
grafo.add_node("juiz", juiz_node)

# Construção das arestas
grafo.add_edge(START, "guardrail") # inicial

# Rota do guardrail
grafo.add_conditional_edges(
    "guardrail",
    decide_after_guardrail,
    {"roteador": "roteador","end": END,},
)

# Decidindo a rota que vai seguir
grafo.add_conditional_edges(
    "roteador",
    decide_after_router,
    {
        "engenharia": "engenharia",
        "app": "app",
        "dados": "dados",
        "analise_completa": "analise_completa",
        "end": END,
    },
)

# Rotas pós nós especialistas
grafo.add_conditional_edges(
    "engenharia",
    decide_after_specialist,
    {"orquestrador": "orquestrador", "end": END},
)
grafo.add_conditional_edges(
    "app",
    decide_after_specialist,
    {"orquestrador": "orquestrador", "end": END},
)
grafo.add_conditional_edges(
    "dados",
    decide_after_specialist,
    {"orquestrador": "orquestrador", "end": END},
)
grafo.add_conditional_edges(
    "analise_completa",
    decide_after_specialist,
    {"orquestrador": "orquestrador", "end": END},
)

# Rota pós orquestrador
grafo.add_conditional_edges(
    "orquestrador",
    decide_after_orquestrador,
    {"juiz": "juiz", "end": END},
)


grafo.add_edge("juiz", END) # Fim

app = grafo.compile() # Fluxo completo compilado


def executar_fluxo_tria(pergunta_usuario: str, session_id: int) -> str:
    final_state = app.invoke({"input": pergunta_usuario, "session_id": session_id})
    print(memory_store)
    set_history(session_id, memory_store[session_id])
    if final_state.get("erro"):
        return f"Erro: {final_state['erro']}"
    return final_state.get("resposta_usuario", "Não foi possível responder.") # isso é um if não tiver resposta_usuario, mostre a "não foi possivel..."


def Tria(pergunta_usuario, cod_usuario):
    try:
        return executar_fluxo_tria(pergunta_usuario, cod_usuario)
    except Exception as e:
        print("Ocorreu um erro ao consumir a API: ", e)

        if ("quota" in str(e)):
            # Quando ocorrer um erro, vai tentar pegar a outra api caso tenha ultrapassado o limite diário
            api_key = os.getenv("GOOGLE_GEMINI_API_RESERVA")

            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.7,
                top_p=0.95,
                google_api_key=api_key
            )

            llm_fast = ChatGoogleGenerativeAI( 
                model="gemini-2.0-flash", # Modelo baseado em performance
                temperature=0.2, # Modelo deterministico, não vai ser criativo. Vai ser direto para o usuário evitando modificar qualquer coisa
                google_api_key=api_key
            )
        else:
            raise Exception(f"Ocorreu um erro ao consumir a API: {e}")

        try: 
            return executar_fluxo_tria(pergunta_usuario, cod_usuario)
        except Exception as ex:
            raise Exception(f"O limite diário da API do gemini foi ultrapassado ou ocorreu outro erro: {ex}")



# Teste manual da IA sem precisar chamar na API
# while True:
#     usuario = input("\n> ")

#     if usuario in  ("sair", "tchau", "bye"):
#         break

#     resposta = Tria(usuario, 2)

#     print(f"\nIA: {resposta}")