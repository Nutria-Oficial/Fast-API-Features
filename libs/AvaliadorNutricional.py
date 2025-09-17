import pandas as pd

def get_total_by_ingrediente(i:dict) -> float:
    # --------- Pontos negativos ---------
    pontos_negativos = 0

    # Caloria
    nutriente = i["nCaloria"]
    metrica = 80
    nutriente = nutriente//metrica
    nutriente = nutriente if nutriente < 10 else 10
    pontos_negativos += nutriente

    # Açucares Totais
    nutriente = i["nAcucar"]
    metrica = 4.5
    nutriente = nutriente//metrica
    nutriente = nutriente if nutriente < 10 else 10
    pontos_negativos += nutriente

    # Gordura Saturada
    nutriente = i["nGorduraSaturada"]
    metrica = 1
    nutriente = nutriente//metrica
    nutriente = nutriente if nutriente < 10 else 10
    pontos_negativos += nutriente

    # Sódio
    nutriente = i["nSodio"]
    metrica = 90
    nutriente = nutriente//metrica
    nutriente = nutriente if nutriente < 10 else 10
    pontos_negativos += nutriente


    # --------- Pontos Positivos ---------
    pontos_positivos = 0

    # Fibras
    nutriente = i["nFibra"]
    metrica = 0.7
    nutriente = nutriente//metrica
    nutriente = nutriente if nutriente < 5 else 5
    pontos_positivos += nutriente

    # Proteina
    nutriente = i["nProteina"]
    metrica = 1.6
    nutriente = nutriente//metrica
    nutriente = nutriente if nutriente < 5 else 5
    pontos_positivos += nutriente


    # --------- Score Final ---------
    score_final = pontos_negativos - pontos_positivos

    return score_final

def pegar_ingredientes_formatados(tabela_mongo:dict) -> dict:
    
    tabela = tabela_mongo.copy()

    model = {
        "nCaloria":0, 
        "nProteina":0, 
        "nAcucar":0, 
        "nFibra":0, 
        "nGorduraSaturada":0, 
        "nSodio":0
    }

    nutrientes_usados = ["Valor Calórico (kcal)", "Proteína (g)", "Açúcar Total (g)", "Fibra Alimentar (g)", "Gordura Saturada (g)", "Sódio (mg)"]

    transform_colunas = {
        "Valor Calórico (kcal)":"nCaloria",
        "Proteína (g)":"nProteina", 
        "Açúcar Total (g)":"nAcucar", 
        "Fibra Alimentar (g)":"nFibra", 
        "Gordura Saturada (g)":"nGorduraSaturada", 
        "Sódio (mg)":"nSodio"
    }

    total = tabela.pop("nTotal")
    valores_inuteis = tabela.pop("_id"), tabela.pop("nCdProduto"), tabela.pop("cNmTabela"), tabela.pop("nPorcao"), tabela.pop("cUnidadeMedida"), tabela.pop("lIngredientes")
    

    df = pd.DataFrame(tabela)

    df = df[df["lNutrientes"].isin(nutrientes_usados)]

    tabela = df.to_dict("split")["data"]

    for i in tabela:
        chave, valor = i[0], i[1]
        valor = valor/total*100 # Deixando nos 100g
        model[transform_colunas[chave]] = valor
    
    return model

def classificar(tabela_nutricional:dict):

    ingrediente = pegar_ingredientes_formatados(tabela_nutricional)

    score = get_total_by_ingrediente(ingrediente)
    
    if (score >= 19):
        classificacao = "E"
    elif (score >= 11):
        classificacao = "D"
    elif (score >= 3):
        classificacao = "C"
    elif (score >= 0):
        classificacao = "B"
    else:
        classificacao = "A"

    return classificacao

