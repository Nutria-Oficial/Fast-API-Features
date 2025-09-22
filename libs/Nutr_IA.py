# Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType
import google.generativeai as genai
import os
from dotenv import load_dotenv
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from urllib.parse import quote_plus
from libs.Exception import Http_Exception

load_dotenv() # Obtendo as variáveis seguras

# Instanciando conexão com o MongoDB
username = quote_plus(os.getenv("MONGO_USER"))

password = quote_plus(os.getenv("MONGO_PWD")) 

uri = f"mongodb+srv://{username}:{password}@nutriamdb.zb8v6ic.mongodb.net/?retryWrites=true&w=majority"

client = MongoClient(uri, server_api=ServerApi('1'))

NutriaMDB = client["NutriaMDB"]

coll_chat = NutriaMDB["chat"]


def desserializar_mensagem(conteudo, tipo):
    """
    Responsável por converter o conteudo de texto da mensagem, para seu objeto equivalente.

    Caso seja **Humano** -> HumanMessage

    Caso seja **Chat**   -> SystemMessage
    """
    if tipo == "user":
        return HumanMessage(content=conteudo)
    elif tipo == "chat":
        return SystemMessage(content=conteudo)


def desserializar_mensagens(data:dict):
    """
    Função que orquestra a utilização da função **desserializar_mensagem()**, para utilizar para todas as mensagens do usuario e do bot em ordem. 
    """

    usuario_mensagens = list(data["lUser"])
    chat_mensagens = list(data["lBot"])

    qtd_trocas_de_mensagem = len(chat_mensagens)
    mensagens = []

    for i in range(qtd_trocas_de_mensagem):
        mensagens.append(desserializar_mensagem(usuario_mensagens[i], "user"))
        mensagens.append(desserializar_mensagem(chat_mensagens[i], "chat"))

    return mensagens
    

def atualizar_chat(memoria:dict, ja_existe:bool):
    """
    ## Função responsável por inserir a memória caso ela não exista ou atualiza-la
    """

    try:
        if (ja_existe):
            coll_chat.update_one({"_id":memoria["_id"]},{"$set": {"lBot": memoria["lBot"]}})
            coll_chat.update_one({"_id":memoria["_id"]},{"$set": {"lUser": memoria["lUser"]}})
        
        else:
            coll_chat.insert_one(memoria)

    except Exception as e:
        raise Http_Exception(500,f"Ocorreu um ERRO ao inserir a memória de código {memoria['_id']}: \n {str(e)}")


def nutria_bot(nCdUsuario:int, iChat:int, pergunta:str) -> str:
    """
    ## Modelo de IA que é responsável por responder perguntas relacionadas a engenharia de alimentos e legislação

    ## Parâmetros:
    - *nCdUsuario*: Código do usuário que está conversando com o modelo
    - *iChat*: Chat que o usuário está. (Um usuário pode ter vários chats)
    - *iPergunta*: A pergunta ou pedido que o usuário mandou

    ## Retorna:
    Retorna uma **string** contendo a resposta do modelo para a pergunta do usuário
    """

    # Configuração básica do Gemini
    api_key = os.getenv("GEMINI_API_KEY")
    genai.configure(api_key=api_key)

    chat = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.7,
        google_api_key=api_key
    )


    # Configuração do prompt
    system_prompt_text = '''
    #Contexto
    Você é um engenheiro de alimentos com conhecimento na legislação completa

    #Missao
    Sua missão é compreender e analisar as perguntas dos usuários para que possa responde-las de acordo com os seus conhecimentos em engenharia de alimentos e na legislação que engloba os alimentos

    #Instruções
    - Leia atentamente a pergunta do usuário e compreenda em detalhes
    - Ligue essa pergunta com algum dado de seu conhecimento
    - Caso não encontre nenhuma informação relevante, indique que você não sabe e onde possívelmente o usuário poderia buscar
    - Responda como um profissional especializado
    - No final de cada pergunta você deve se disponibilizar para responder qualquer outra duvida do usuário
    - Você apenas irá responder perguntas sobre engenharia de alimentos e a legislação

    Pergunta do usuário: {input}
    '''
    
    system_prompt = PromptTemplate.from_template(system_prompt_text)


    # Configuração da memória
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    memoria_list = coll_chat.find({"nCdUsuario":nCdUsuario, "iChat":iChat}).to_list()

    ja_existe = len(memoria_list) > 0

    if (ja_existe):
        memoria = memoria_list[0]
        memoria["lUser"].append(pergunta)

    else:
        memoria = {
            "_id":coll_chat.count_documents({})+1,
            "nCdUsuario":nCdUsuario,
            "iChat":iChat,
            "lUser":[pergunta],
            "lBot":[]
        }

    memory.chat_memory.messages = desserializar_mensagens(memoria)


    # Criação do agente
    agent = initialize_agent(
        llm=chat,
        tools=[],
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=False,
        agent_kwargs={
            "prefix": system_prompt
        }
    )


    # Pergunta do usário e resposta do agente
    resposta = agent.invoke(pergunta)["output"]


    # Criação do juiz
    juiz = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.3,
        google_api_key=api_key
    )

    prompt_juiz = '''
    Você é um avaliador imparcial. Sua tarefa é revisar a resposta de um engenheiro de alimentos.

    Critérios:
    - A resposta está tecnicamente correta?
    - Está clara para o nível médio técnico?
    - O próximo passo sugerido está bem formulado?

    Regras:
    - Não opine ou fale oque está fazendo, apenas replique ou proponha uma versão melhor

    Se a resposta for boa, replique a resposta, sem mais comentários.
    Se tiver problemas, proponha uma versão melhorada, sem mais comentários, apenas a versão melhorada.
    '''


    # Avaliação da resposta
    def avaliar_resposta(pergunta, resposta_tutor):
        mensagens = [
            SystemMessage(content=prompt_juiz),
            HumanMessage(content=f"Pergunta do aluno: {pergunta}\n\nResposta do tutor: {resposta_tutor}")
        ]
        return juiz.invoke(mensagens).content

    avaliacao = avaliar_resposta(pergunta, resposta)


    # Armazenando a resposta do Nutria Bot
    memoria["lBot"].append(avaliacao)

    # Inserindo ou atualizando no banco
    atualizar_chat(memoria, ja_existe)
    
    # Resposta final
    return avaliacao
