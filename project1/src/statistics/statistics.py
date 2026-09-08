import math
import numpy as np

def RankedSearch(query, vectorizedDocs):
    ranking = []
    for nome, vetor in vectorizedDocs.items():
        score = np.dot(query, vetor)
        ranking.append((score, nome))

    ranking.sort(reverse=True)

    return ranking
    

def TF_IDF(indexInvertido, totalDocuments):
    result = {}
    for palavra, valores in indexInvertido.items():
        numDoc = len(valores)

        for documento in valores:
            if (documento[1]==0):
                numDoc = numDoc - 1

        for documento in valores:
            nome_documento = documento[0]
            frequencia = documento[1]
            if nome_documento not in result:
                result[nome_documento] = np.array()

            result[nome_documento] = np.append(result[nome_documento], 
            TermFrequency(frequencia, numDoc)*InverseDocumentFrequency(numDoc, totalDocuments))
            
    return result

def TermFrequency(termRepetition, numberDocumentsHasTerm):
    return termRepetition/numberDocumentsHasTerm

def InverseDocumentFrequency(numberDocumentsHasTerm, totalDocuments):
    return math.log(totalDocuments/numberDocumentsHasTerm)
