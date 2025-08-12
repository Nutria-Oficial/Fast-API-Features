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


def classificar(ingrediente:dict):

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

    return classificacao, score

