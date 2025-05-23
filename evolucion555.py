# PAQUETERIA 

#import pandas as pd

from matplotlib import pyplot as plt
import statistics as st
import imageio.v2 as imageio

import pickle
import os

import numpy as np

from math import cos, sin, radians, atan2, degrees, floor, sqrt

from random import random, randint
from collections import Counter
import time

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.initializers import GlorotNormal

import networkx as nx
from networkx.algorithms.community.quality import modularity

np.random.seed(2024)

# Función para guardar las listas de una generación específica
def guardar_listas_generacion(generacion, depredadores, presas, evolucionpoblaciones, resumengenD, resumengenP, folder_path):
    with open(os.path.join(folder_path, f"evolucionpoblaciones_{generacion}.pkl"), "wb") as f:
        pickle.dump(evolucionpoblaciones, f)
    
    with open(os.path.join(folder_path, f"resumengenD_{generacion}.pkl"), "wb") as f:
        pickle.dump(resumengenD, f)
    
    with open(os.path.join(folder_path, f"resumengenP_{generacion}.pkl"), "wb") as f:
        pickle.dump(resumengenP, f)

    with open(os.path.join(folder_path, f"depredadores_{generacion}.pkl"), "wb") as f:
        pickle.dump(depredadores, f)
    
    with open(os.path.join(folder_path, f"presas_{generacion}.pkl"), "wb") as f:
        pickle.dump(presas, f)

# Función para cargar las listas de una generación específica
def cargar_listas_generacion(generacion, folder_path):
    with open(os.path.join(folder_path, f"evolucionpoblaciones_{generacion}.pkl"), "rb") as f:
        evolucionpoblaciones = pickle.load(f)
    
    with open(os.path.join(folder_path, f"resumengenD_{generacion}.pkl"), "rb") as f:
        resumengenD = pickle.load(f)
    
    with open(os.path.join(folder_path, f"resumengenP_{generacion}.pkl"), "rb") as f:
        resumengenP = pickle.load(f)
        
    with open(os.path.join(folder_path, f"depredadores_{generacion}.pkl"), "rb") as f:
        depredadores = pickle.load(f)
    
    with open(os.path.join(folder_path, f"presas_{generacion}.pkl"), "rb") as f:
        presas = pickle.load(f)    
    
    return depredadores, presas, evolucionpoblaciones, resumengenD, resumengenP

# Determinar la última generación completada
def obtener_ultima_generacion(folder_path):
    ultima_generacion = 0
    for filename in os.listdir(folder_path):
        if filename.startswith("evolucionpoblaciones_") and filename.endswith(".pkl"):
            generacion = int(filename.split("_")[1].split(".")[0])
            ultima_generacion = max(ultima_generacion, generacion)
    return ultima_generacion

def generacion0 (numeroP):# PARA COMENZAR, EN LA GENERACION 0, 
    # poblar el entorno de redes neuronales de decision de la clase presa
    presas=[]
    for P in range(numeroP): 
       # las presas heredan los atributos y los grados de libertad de los padres
        atributos = [] # valor de los inputs: los no seleccionados a 0
        atributosP = []  # inputs a considerar en la red neuronal: numatritutos valores entre 16
        exitostrainP = []  # lista de valores de angulo y distancia que en los juegos han resultado buenas estrategias
        exitostestP = [] # lista de angulos y distancias que en la huida han funcionado
        exitostestRP = []
        tiemposP = [] # tiempo de la red neuronal en tomar una decision, un valor por cada decision de caza
        presa=[P,
               np.random.uniform(x_min,x_max),# posicion aleatoria de cada presa
               np.random.uniform(y_min,y_max),
               np.random.uniform(0,drmaxP),# angulo
               np.random.uniform(0,velocidadP),# velocidad
               [],# 5, resultados de los conjuntos de juegos entre amigotes
               [], # 6, contador de eventos caza
               0, # 7, contador de encuentros reproductivos           
               1, # 8, fitness 
               0, # 9, edad
               exitostestP, # 10, velocidad y angulo de exitos de huida en la caza          
               exitostestRP, # 11, exitos reproductivos (las presas siempre que se encuentran, ligan)
               exitostrainP, # 12, tabla de angulos y velocidades con exito en la huida (entrenamiento) 
               atributosP, # 13, seleccion de los inputs 
               [], # 14, entrenamientos
               0, # 15, contador de conexiones
               # tupla de maximo 16 que son los atributos de cada presa
               atributos, # 16, valores de los inputs, se comportará según un input de longitud len(atributosP)
               numprocesadores,
               [], # tabla de valores de atributos utilizada en cada juego          
               [], # Inicializacion entre la entrada y la capa oculta           
               [], # Inicialización de los pesos de las conexiones dentro de la capa oculta 
               [], # Inicializacion de los pesos entre la capa oculta y el output 
               0, # tiempo de proceso para tomar la decision
               0, # 23, agilidad
               0, # 24, agresividad          
               0, # 25, cocea
               0, # 26, embiste  
               0, # 27, tamaño de sus defensas (cuernos, espinas, veneno, camuflaje,...)
               0, # 28, se camufla
               0, # 29, apesta o escupe
               0, # 30, grita 
               0,  
               0, 
               0,          
               0, 
               0,   
               0, 
               0, 
               0, 
               tiemposP,
               None
               ]
        presas.append(presa)
        # Elegir el numatributos inicial
        presa[13] = np.random.choice(np.arange(23,39), numatributos, replace=False).tolist()
      
        # Establecer a 0 los campos no seleccionados en atributosP y a su valor en 16
        for i in range(23,39):  # Campos entre 23 y 38 (ambos inclusive)
            if i not in presa[13]:
                presa[i] = 0
            else:
                presa[i] = np.random.uniform(0,0.5)
            atributos.append(presa[i]) # vector de valores de los atributos en 16
    
    # poblar el entorno de depredadores con redes neuronales de pesos aleatorios
    depredadores=[] # lista de listas depredador
    for D in range(numeroD):
        # VALORES INICIALES: generacion 0, tiempo 0
        # la escala de valor y la estructura de la red son heredable, no los pesos
        
        atributosD = [] # lista vacía con un numero variable de entradas por su posicion
        atributos = [] # lista de valores de las entradas
        exitostrainD = [] # lista vacia de dos valores con un numero variable de entradas
        exitostestD = []
        exitostestRD = []
        tiemposD = [] # lista vacia de un valor con un numero variable de entradas
        depredador = [D,# cada depredador tiene su lista de valores iniciales
                      np.random.uniform(x_min,x_max),
                      np.random.uniform(y_min,y_max),# posicion aleatoria de cada depredador
                      np.random.uniform(0,drmaxD),# angulo
                      np.random.uniform(0,velocidadD),# velocidad
                      [],# conjunto de juegos entre dos amigos que comparten intereses: experiencias y fracasos
                      [], # 6, contador de eventos de caza
                      0, # 7, contador de eventos reproductivos                  
                      reglatrofica, # 8, fitness o gordura                  
                      0, # 9, edad 
                      exitostestD, # 10, tabla de angulos y velocidades con exito de caza                   
                      exitostestRD, # 11, tabla de angulos y velocidades con exitos reproductivos
                      exitostrainD, # 12, tabla de angulos y velocidades con exito (entrenamiento)
                      atributosD, # 13, inputs que se van a considerar != 0
                      [], # 14, contador de entrenamientos
                      0, # 15, contador de conexiones
                      atributos, # 16, valores de los inputs elejidos entre 23 a 39 
                                  # (los no seleccionados en atributosD, son =0)
                      numprocesadores, # 17, dos hidden layers iguales que simulan una recurrente
                      [], # 18, lista de inputs para el aprendizaje
                      [], # Inicializacion entre la entrada y la capa oculta  
                      [], # Inicializacion dentro de la capa oculta
                      [], # Inicialización de los pesos de las conexiones entre la capa oculta y la capa de salida
                      0, # 22, tiempo de proceso para tomar la decision
                      0, # 23, agilidad
                      0, # 24, agresividad                   
                      0, # 25, agarra
                      0, # 26, muerde 
                      0, # 27, tamaño de garras y dientes
                      0, # 28, salta
                      0, # 29, al cuello
                      0, # 30, ruge y acojona 
                      0,
                      0,
                      0,                   
                      0, 
                      0,
                      0, 
                      0, 
                      0,
                      tiemposD, # 39, tabla de tiempos de cada decision neuronal
                      None
                      ] #generacion y paso de tiempo 
        depredadores.append(depredador)
        
        # Elegir el numatributos inicial
        depredador[13] = np.random.choice(np.arange(23,39), numatributos, replace=False).tolist()
      
        # Establecer a 0 los campos no seleccionados en atributosD
        for i in range(23,39):  # Campos entre 23 y 38 (inclusive)
            if i not in depredador[13]:
                depredador[i] = 0
            else:
                depredador[i] = np.random.uniform(0.5,1)
            atributos.append(depredador[i]) # vector del valor de los inputs en 16
    
    # GRAFICO (llama a la funcion dibujar) SITUACION INICIAL GEN=0, T=0 
    dibujarcaza(depredadores,presas,generacion=0,tiempo=0)    # devuelve grafico png 
    return (depredadores,presas)

# Función para normalizar los datos
def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def prune(model, threshold=0.1):
    pruned_weights = []
    for layer in model.layers:
        if isinstance(layer, Dense):
            weights = layer.get_weights()
            if len(weights) > 0:
                pruned_weights.append(np.where(np.abs(weights[0]) >= threshold, weights[0], 0.0))
                pruned_weights.append(np.where(np.abs(weights[1]) >= threshold, weights[1], 0.0))
    
    return pruned_weights

def neuralD(depredador):
    # si el depredador viene de una generacion anterior, no aprende mas
    if depredador[40] == None: # si ya tiene experiencia anterior
        # tambien puede haber pollos que no hayan acumulado experiencia pero no se hayan descartado para evitar demasiadas bajas
        # estos quedan como cachorros para otra oportunidad en la siguiente generacion
        if len(depredador[18]) != 0:
            # Obtener datos de entrada y salida para el entrenamiento
            train_inputs = np.array(depredador[18])
            train_outputs = np.array(depredador[12])
            # Convierte tus listas de Python a tensores
            train_inputs = tf.convert_to_tensor(train_inputs, dtype=tf.float32)
            train_outputs = tf.convert_to_tensor(train_outputs, dtype=tf.float32)
       
            # Crear un modelo secuencial
            modeloD = Sequential()
            # Capa de entrada con el número de atributosD
            modeloD.add(Input(shape=(len(depredador[18][0],)), name="input_layer", dtype='float32'))
    
            # Capa hidden numprocesadoresD inicialización Xavier solo la primera vez 
            pesosinput = np.array(depredador[19], dtype=np.float32)
            if pesosinput.size == 0:  # Verificar si los pesos están vacíos
                modeloD.add(Dense(depredador[17], activation='relu', kernel_initializer=GlorotNormal(), name="hidden_layer"))
            # cuando ya tiene pesos, la inicializacion es con la experiencia anterior
            else:
                modeloD.add(Dense(depredador[17], activation='relu', name="hidden_layer"))
                modeloD.layers[-1].set_weights([depredador[19], modeloD.layers[-1].get_weights()[1]])  # incializa con pesos y sesgos previos
        
            # Segunda capa oculta para simular una capa recurrente, todos con todos
            pesoshidden = np.array(depredador[20], dtype=np.float32)
            if pesoshidden.size == 0:  # Verificar si los pesos están vacíos
                modeloD.add(Dense(depredador[17], activation='relu', kernel_initializer=GlorotNormal(), name="recurrent_layer"))
            else:
                modeloD.add(Dense(depredador[17], activation='relu', name="recurrent_layer"))
                modeloD.layers[-1].set_weights([depredador[20],modeloD.layers[-1].get_weights()[1]])  # inicializa con pesos y sesgos previos
        
            # Capa output con dos nodos: angulo y velocidad
            pesosoutput = np.array(depredador[21], dtype=np.float32)
            if pesosoutput.size == 0:  # Verificar si los pesos están vacíos
                modeloD.add(Dense(2, activation='linear', kernel_initializer=GlorotNormal(), name="output_layer"))
            else:
                modeloD.add(Dense(2, activation='linear', name="output_layer"))
                modeloD.layers[-1].set_weights([depredador[21], modeloD.layers[-1].get_weights()[1]])  # inicializa con pesos y sesgos previos
        
            # Compilar el modelo
            modeloD.compile(loss='mean_squared_error', optimizer='adam')    
            
            # Configurar EarlyStopping
            early_stopping_monitor = EarlyStopping(
                monitor='val_loss',  # Monitorear la pérdida de validación
                patience=10,         # Detener el entrenamiento después de 10 épocas sin mejora
                restore_best_weights=True  # Restaurar los pesos del modelo a la mejor época
            )
            
            # Medir el tiempo de inicio
            start_time = time.process_time()
            
            numjuegos=len(train_inputs)
            # Entrenamiento con EarlyStopping
            history = modeloD.fit(train_inputs, train_outputs, validation_split=0.2, epochs=100, callbacks=[early_stopping_monitor], verbose=0)
            
            # Medir el tiempo de finalización
            end_time = time.process_time()
            # Calcular la duración del entrenamiento
            tiemporeaccionD = (end_time - start_time)/numjuegos
            
            utilizadas = len(history.history['loss'])
            if 'val_loss' in history.history:
                utilizadasD = min(utilizadas, len(history.history['val_loss']))
            
            #Plot del historial de pérdida
            plt.plot(history.history['loss'], 'o-', color='black', linewidth=2, label='training')      # puntos + línea negra
            plt.plot(history.history['val_loss'], '-', color='black', linewidth=2, label='validation') # línea negra
            plt.xlabel('epochs')
            plt.ylabel('loss')
            plt.legend()
            plt.title('Training & Validation Loss')
            plt.show()
            
            # Podar pesos y sesgos 
            pruned_weights = prune(modeloD, threshold=0.1)
            depredador[19] = pruned_weights[0]
            depredador[20] = pruned_weights[2]
            depredador[21] = pruned_weights[4]     
    
            depredador[40] = modeloD
            
            # Calcular la modularidad entre las capas ocultas
            if pesoshidden.size != 0:
                # Crear un grafo dirigido a partir de la matriz de pesos específicos entre las capas ocultas
                G = nx.DiGraph(pesoshidden)
                # Calcular las comunidades utilizando un algoritmo de detección de comunidades
                comunidades = nx.algorithms.community.greedy_modularity_communities(G)
                # Calcular la modularidad
                modularidad = nx.algorithms.community.quality.modularity(G, comunidades)
                # Guardar el valor de modularidad en presa[15]
                depredador[15] = modularidad
            else:
                # No se pueden calcular los pesos específicos entre las capas ocultas
                depredador[15] = None

    return depredador, tiemporeaccionD, utilizadasD

def neuralP(presa):  
    if presa[40] == None:  
        if len(presa[18]) != 0:
            train_inputs = np.array(presa[18])
            train_outputs = np.array(presa[12])
            train_inputs = tf.convert_to_tensor(train_inputs, dtype=tf.float32)
            train_outputs = tf.convert_to_tensor(train_outputs, dtype=tf.float32)
           
            modeloP = Sequential()
            modeloP.add(Input(shape=(len(presa[18][0],)), name="input_layer", dtype='float32'))
    
            pesosinput = np.array(presa[19], dtype=np.float32)
            if pesosinput.size == 0:  
                modeloP.add(Dense(presa[17], activation='relu', kernel_initializer=GlorotNormal(), name="hidden_layer"))
            else:
                modeloP.add(Dense(presa[17], activation='relu', name="hidden_layer"))
                modeloP.layers[-1].set_weights([presa[19], modeloP.layers[-1].get_weights()[1]])  
        
            pesoshidden = np.array(presa[20], dtype=np.float32)
            if pesoshidden.size == 0:  
                modeloP.add(Dense(presa[17], activation='relu', kernel_initializer=GlorotNormal(), name="recurrent_layer"))
            else:
                modeloP.add(Dense(presa[17], activation='relu', name="recurrent_layer"))
                modeloP.layers[-1].set_weights([presa[20], modeloP.layers[-1].get_weights()[1]])  
        
            pesosoutput = np.array(presa[21], dtype=np.float32)
            if pesosoutput.size == 0:  
                modeloP.add(Dense(2, activation='linear', kernel_initializer=GlorotNormal(), name="output_layer"))
            else:
                modeloP.add(Dense(2, activation='linear', name="output_layer"))
                modeloP.layers[-1].set_weights([presa[21], modeloP.layers[-1].get_weights()[1]])  
        
            modeloP.compile(loss='mean_squared_error', optimizer='adam')
            batch_size=10
            
            # Definir EarlyStopping
            early_stopping_monitor = EarlyStopping(patience=5, monitor='val_loss', mode='min', verbose=0)
            start_time = time.process_time()
            # Entrenar con EarlyStopping
            history = modeloP.fit(train_inputs, train_outputs, validation_split=0.2, epochs=100, batch_size=batch_size, callbacks=[early_stopping_monitor], verbose=0)
    
            end_time = time.process_time()
            tiemporeaccionP = (end_time - start_time) / len(train_inputs)
            
            # Poda con OBC
            pruned_weights = prune(modeloP, threshold=0.1)
            presa[19] = pruned_weights[0]
            presa[20] = pruned_weights[2]
            presa[21] = pruned_weights[4]
            
            # Obtener el número de experiencias utilizadas hasta EarlyStopping
            utilizadas = len(history.history['loss'])
            if 'val_loss' in history.history:
                utilizadasP = min(utilizadas, len(history.history['val_loss']))
            
            presa[40] = modeloP
            
            # Calcular la modularidad entre las capas ocultas
            if pesoshidden.size != 0:
                # Crear un grafo dirigido a partir de la matriz de pesos específicos entre las capas ocultas
                G = nx.DiGraph(pesoshidden)
                # Calcular las comunidades utilizando un algoritmo de detección de comunidades
                comunidades = nx.algorithms.community.greedy_modularity_communities(G)
                # Calcular la modularidad
                modularidad = nx.algorithms.community.quality.modularity(G, comunidades)
                # Guardar el valor de modularidad en presa[15]
                presa[15] = modularidad
            else:
                # No se pueden calcular los pesos específicos entre las capas ocultas
                presa[15] = None

    return presa, tiemporeaccionP, utilizadasP

def caza(depredador, presa):
    intcomun = list(np.intersect1d(depredador[13], presa[13]))
    exito = None  # inicializamos
    # se puede hacer todo lo complicado que se quiera, por eso necesitamos una funcion aparte
    # incorporar atributos puede hacer mas lenta la decision, asi que para que merezca la pena, debe haber una presion en contra
    # solo aumentaran atributos si el tiempo de decision lo admite, esto a la vez presiona hacia una mejor eficiencia
    if abs(depredador[22] - presa[22]) < depredador[22] * 0.5: # si uno tarda demasiado en decidir, gana el otro: para seleccionar eficiencia en la decision
        if intcomun:
            for i in range(23,39):  # Campos entre 23 y 38 (inclusive) 
                if exito is not None:  # Salir del bucle si exito ya tiene un valor
                    break
                
                if presa[i] != 0 and depredador[i] != 0:
                    if depredador[i] > presa[i]:
                        exito = True # la presa es cazada a pesar de ser mas eficiente
                        # al hacerlo secuencial damos mas importancia al primer atributo común que al segundo, y sucesivamente
                        # Obtenemos la posición del atributo en el vector 13 del depredador
                        posD = depredador[13].index(i)
                        if posD != 0: # salvo que este el primero, el atributo promociona en prioridad
                            depredador[13][posD], depredador[13][posD-1] = depredador[13][posD-1], depredador[13][posD]
                        
                        depredador[i] += 1 / pasostiempo
                        depredador[8] += presa[8] / reglatrofica # aprovecha la energía trofica al completo
                        
                        break # se la come y sale del bucle
                        
                    else: # al tener un atributo de mayor prioridad mayor, la presa escapa
                        # premio a la presa y castigo al depredador, adicional 
                        exito = False # no caza
                        depredador[i] += -1 / pasostiempo #castigamos al depredador
                        presa[i] += 1 / (pasostiempo*reglatrofica) # promocionamos a la presa que ha huido
                        # promociona el atributo con el que logrado escapar
                        posD = depredador[13].index(i)
                        posP = presa[13].index(i)
                        if posP != 0:
                            presa[13][posP],presa[13][posP-1] = presa[13][posP-1],presa[13][posP]
                        # no directamente, pero triunfar implica privilegios territoriales ante sus semejantes
                        presa[8] += depredador[8] / reglatrofica                         
                        # para que no salgan valores del rango entre 0 y 1:
                        if depredador[i] > 1: depredador[i] = 1
                        if depredador[i] < 0: depredador[i] = 0
                        if presa[i] > 1: presa[i] = 1
                        if presa[i] < 0: presa[i] = 0
        
                        # la presa huye a otro lugar aleatorio
                        presa[1] = np.random.uniform(x_min,x_max)
                        presa[2] = np.random.uniform(y_min,y_max)
                        # como ya ha huido, sale del bucle
                        break
                
                if exito is not None:  # Salir del bucle si exito ya tiene un valor
                    break # si salen del bucle, hay que salir por fases
                    
        else: # si no comparten ningun atributo comun, el grande se come al chico 
            if depredador[8] > presa[8]:
                exito = True
                depredador[8] += presa[8] / reglatrofica
            else:
                exito = False
                presa[8] += depredador[8] / pasostiempo  # por las molestias
                presa[1] = np.random.uniform(x_min,x_max)
                presa[2] = np.random.uniform(y_min,y_max) 
    else: # si la presa es mas rapida, huye y no hay enfrentamiento ni riesgo
        exito = False
        presa[8] += depredador[8] / pasostiempo
        
    return depredador, presa, exito

def decisionD(depredador):
    # mediremos el tiempo que tarda la red neuronal en tomar una decision
    inicio_tiempo = time.process_time()
    
    datos_prediccion = np.array([depredador[16]])  # atributos almacenados en el depredador
    predicciones = depredador[40].predict(datos_prediccion)

    tiempoD = time.process_time() - inicio_tiempo

    # Desnormalizar las predicciones si es necesario
    alphaD = predicciones[0,0]  
    carreraD = predicciones[0,1]  

    return alphaD, carreraD, tiempoD

def decisionP(presa): # viene a ser el entorno de test 
    # Hacer predicciones y medir el tiempo
    inicio_tiempo = time.process_time()
    datos_prediccion = np.array([presa[16]])  # atributosP almacenados en presa
    predicciones = presa[40].predict(datos_prediccion)
    
    tiempoP = time.process_time() - inicio_tiempo

    # Desnormalizar las predicciones si es necesario
    alphaP = predicciones[0,0]  
    carreraP = predicciones[0,1]  

    return alphaP, carreraP, tiempoP

def contarno0D(depredadores):
    # Sustituimos los valores de 0.01 por 0 y los valores de 0.99 por 1
    for depredador in depredadores:
        for i in range(16):
            if depredador[i] == 0.01:
                depredador[i] = 0
            elif depredador[i] == 0.99:
                depredador[i] = 1
    
    # Inicializamos
    cantidadno0D = [0] * 16   
    sumasno0D = [0] * 16 
    mediasno0D = [0] * 16
    
    # Aplanamos la lista de depredadores para evitar listas anidadas
    depredadoresD = depredadores

    # Recorremos los 16 elementos de los vectores
    for i in range(16):
        # Calculamos la suma de valores no cero en la posición i
        sumasno0D[i] = sum(depredadoresD[j][i] for j in range(len(depredadoresD)) if isinstance(depredadoresD[j], list) and isinstance(depredadoresD[j][i], int) and depredadoresD[j][i] != 0)
        
        # Contamos el número de valores no cero en la posición i
        cantidadno0D[i] = sum(1 for j in range(len(depredadoresD)) if isinstance(depredadoresD[j], list) and isinstance(depredadoresD[j][i], int) and depredadoresD[j][i] != 0)

    # Calculamos las medias de valores no cero en cada posición
    for i in range(16):
        if cantidadno0D[i] != 0:  # Verificamos que haya al menos un valor no cero en la posición i
            mediasno0D[i] = sumasno0D[i] / cantidadno0D[i]
    
    return cantidadno0D, mediasno0D

def contarno0P(presas):
    # Sustituimos los valores de 0.01 por 0 y los valores de 0.99 por 1
    for presa in presas:
        for i in range(16):
            if presa[i] == 0.01:
                presa[i] = 0
            elif presa[i] == 0.99:
                presa[i] = 1
    # Inicializamos
    cantidadno0P = [0] * 16   
    sumasno0P = [0] * 16  
    mediasno0P = [0] * 16
    # Aplanamos la lista de depredadores para evitar listas anidadas
    presasP = presas
    # Recorremos los 16 elementos de los vectores
    for i in range(16):
        # Calculamos la suma de valores no cero en la posición i
        sumasno0P[i] = sum(presasP[j][i] for j in range(len(presasP)) if isinstance(presasP[j], list) and isinstance(presasP[j][i], int) and presasP[j][i] != 0)
        # Contamos el número de valores no cero en la posición i
        cantidadno0P[i] = sum(1 for j in range(len(presasP)) if isinstance(presasP[j], list) and isinstance(presas[j][i], int) and presasP[j][i] != 0)

    # Calculamos las medias de valores no cero en cada posición
    for i in range(16):
        if cantidadno0P[i] != 0:  # Verificamos que haya al menos un valor no cero en la posición i
            mediasno0P[i] = sumasno0P[i] / cantidadno0P[i]
    
    return cantidadno0P, mediasno0P

def hijo(atributos_padre1, atributos_padre2):
    # Calcular la media de la cantidad de atributos de ambos padres
    media = (len(atributos_padre1) + len(atributos_padre2)) / 2
    
    # Elegir aleatoriamente entre el entero menor o mayor más cercano a la media
    num_atributos_hijo = int(media)
    if randint(0,1) == 1:
        num_atributos_hijo += 1
    
    # Inicializar los índices para recorrer los atributos de los padres
    index_padre1 = 0
    index_padre2 = 0
    
    # Lista para almacenar los atributos del hijo
    atributos_hijo = []
    
    # Comenzar a construir los atributos del hijo considerando las prioridades de los padres
    for i in range(num_atributos_hijo):
        # Alternar entre los padres para seleccionar los atributos
        if i % 2 == 0:
            # Si hay atributos disponibles en el padre1 y no están en el hijo, añadirlos
            while index_padre1 < len(atributos_padre1) and atributos_padre1[index_padre1] in atributos_hijo:
                index_padre1 += 1
            if index_padre1 < len(atributos_padre1):
                atributos_hijo.append(atributos_padre1[index_padre1])
                index_padre1 += 1
        else:
            # Si hay atributos disponibles en el padre2 y no están en el hijo, añadirlos
            while index_padre2 < len(atributos_padre2) and atributos_padre2[index_padre2] in atributos_hijo:
                index_padre2 += 1
            if index_padre2 < len(atributos_padre2):
                atributos_hijo.append(atributos_padre2[index_padre2])
                index_padre2 += 1
    
    return atributos_hijo

def mutacion (atributos, numprocesadores):
    # la probabilidad debe ser simetrica en mas o menos, y sera la presion selectiva la que promueva mas uno u otro sentido              
    mutacion = random() # la mutacion es simetrica sumando o restando procesador
    if mutacion <= tasamutacion:               
        # añadir o eliminar un nodo procesador en ambas capas hidden (para mantener la simulacion de recurrencia)
        if randint(0, 1) == 1:# elige al azar 0, 1
            numprocesadores += 1
        else:
            if numprocesadores > 5: # por debajo de un minimo es ser tonto del pueblo
                numprocesadores -= 1
                    
    mutacion=random()
    if mutacion <= tasamutacion:   
        # añadir o eliminar uno de los atributos, no hay preferencia
        numatributos=len(list(filter(lambda x: x != 0, atributos)))
        caracruz = randint(0, 1) # moneda al aire: 0 o 1
        if caracruz == 1:
            numatributos += 1
            atributonuevo = randint(23,38) # del 23 al 38, ambos incluidos
            while atributonuevo in atributos: # se cambia hasta encontrar uno no repetido
                atributonuevo = randint(23,38) # para evitar un atributo repetido
            atributos.append(atributonuevo)
        else:
            if numatributos > 1:
                numatributos -= 1
                atributos.pop(randint(0, numatributos)) #se quita un atributo al azar 
            else:
                numatributos = 1

    return atributos, numprocesadores

# preparar datos para las graficas
def lista_a_diccionario(datos):
    
    etiquetas = ['generacion', 'identificador', 'eventos_caza', 'alimentacion', 'eventos_reproductivos', 'edad', 'atributos', 'entrenamientos', 'modularidad', 'procesadores', 'tiempo_proceso', 'improve_with_el_tiempo','worsens_with_el_tiempo','improve_with_el_ejercicio','worsens_with_el_ejercicio','improve_with_el_exito','worsens_with_el_exito','improve_with_la_alimentacion','worsens_with_la_alimentacion','improve_with_la_edad_y_la_alimentacion','worsens_with_la edad_y_la_alimentacion','improve_with_la_edad_y_menos_alimentacion','worsens_with_la_edad_y_menos_alimentacion','improve_with_la_actividad_reproductora','worsens_with_la_actividad_reproductora','improve_with_el_numero_de_intereses','worsens_with_el_numero_de_intereses']
    diccionario = {}

    for i, etiqueta in enumerate(etiquetas):
        diccionario[etiqueta] = datos[i]

    return diccionario

# para que no se salga del ecosistema
def controlfronteraD(depredador):
    
    if depredador[1] > x_max:
        depredador[1] = np.random.uniform(x_min,x_max)
    if depredador[2] > y_max:
        depredador[2] = np.random.uniform(y_min,y_max)
    if depredador[1] < x_min:
        depredador[1] = np.random.uniform(x_min,x_max)
    if depredador[2] < y_min:
        depredador[2] = np.random.uniform(y_min,y_max)
            
def controlfronteraP(presa):
    
    if presa[1] > x_max:
        presa[1] = np.random.uniform(x_min,x_max)
    if presa[2] > y_max:
        presa[2] = np.random.uniform(y_min,y_max)
    if presa[1] < x_min:
        presa[1] = np.random.uniform(x_min,x_max)
    if presa[2] < y_min:
        presa[2] = np.random.uniform(y_min,y_max)    
            
# CREAR GRAFICOS PARA CADA PASO DE TIEMPO EN CADA GENERACION
def dibujarcaza(depredadores, presas, generacion, tiempo):
    fig, ax = plt.subplots(figsize=(9.6, 5.4))

    plt.xlim([x_min * 0.25, x_max * 0.25])
    plt.ylim([y_min * 0.25, y_max * 0.25])
    ax.set_facecolor('#f7f7f7')  # Fondo claro

    # Depredadores
    if depredadores:
        maxfitnessD = max([d[8] for d in depredadores], default=1)
        for depredador in depredadores:
            x, y, angulo, fitness = depredador[1], depredador[2], depredador[3], depredador[8]
            tamano = max(10, 100 * fitness / (maxfitnessD if maxfitnessD != 0 else 1))  # Evita división por cero
            color = (1, 0, 0)  # Rojo para depredadores
            plt.scatter(x, y, s=tamano, c=[color], alpha=0.6, edgecolors='black')

    # Presas
    if presas:
        maxfitnessP = max([p[8] for p in presas], default=1)
        for presa in presas:
            x, y, angulo, fitness = presa[1], presa[2], presa[3], presa[8]
            tamano = max(10, 33 * fitness / (maxfitnessP if maxfitnessP != 0 else 1))
            color = (0, 1, 0)  # Verde para presas
            plt.scatter(x, y, s=tamano, c=[color], alpha=0.6, edgecolors='black', marker='o')

    # Configuración del entorno
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    # Información en el gráfico
    plt.figtext(0.025, 0.95, 'HUNTING', fontsize=12, fontweight='bold')
    plt.figtext(0.025, 0.90, f'GENERATION: {generacion}')
    plt.figtext(0.025, 0.85, f'TIME: {tiempo}')
    plt.figtext(0.025, 0.80, f'Depredator: {len(depredadores)}')
    plt.figtext(0.025, 0.75, f'Prey: {len(presas)}')

    # Guardar imagen
    plt.savefig(f"{generacion}-{tiempo}.png", dpi=100)
    plt.close(fig)
    
def dibujarreproduccion(depredadores, presas, depredadoreshijos, presashijos, generacion, tiempo):
    fig, ax = plt.subplots(figsize=(9.6, 5.4))

    plt.xlim([x_min * 0.25, x_max * 0.25])
    plt.ylim([y_min * 0.25, y_max * 0.25])
    ax.set_facecolor('#f7f7f7')  # Fondo claro
    ax.set_aspect('equal')  # Asegura que los ejes tengan la misma escala
    
    # Dibujar depredadores
    for depredador in depredadores:
        x, y, angulo = depredador[1], depredador[2], depredador[3]
        plt.scatter(x, y, s=50, c='red', edgecolors='black', alpha=0.6)
        #plt.arrow(x, y, np.cos(angulo) * 0.2, np.sin(angulo) * 0.2, head_width=0.1, fc='red', ec='black')
    
    # Dibujar presas
    for presa in presas:
        x, y, angulo = presa[1], presa[2], presa[3]
        plt.scatter(x, y, s=30, c='green', edgecolors='black', alpha=0.6)
        #plt.arrow(x, y, np.cos(angulo) * 0.2, np.sin(angulo) * 0.2, head_width=0.1, fc='green', ec='black')
    
    # Dibujar nuevos depredadores (hijos)
    for hijo in depredadoreshijos:
        x, y, angulo = hijo[1], hijo[2], hijo[3]
        plt.scatter(x, y, s=50, c='darkred', edgecolors='black', alpha=0.6)
        #plt.arrow(x, y, np.cos(angulo) * 0.2, np.sin(angulo) * 0.2, head_width=0.1, fc='darkred', ec='black')
    
    # Dibujar nuevas presas (hijos)
    for hijo in presashijos:
        x, y, angulo = hijo[1], hijo[2], hijo[3]
        plt.scatter(x, y, s=30, c='darkgreen', edgecolors='black', alpha=0.6)
        #plt.arrow(x, y, np.cos(angulo) * 0.2, np.sin(angulo) * 0.2, head_width=0.1, fc='darkgreen', ec='black')
    
    # Información en el gráfico
    plt.figtext(0.025, 0.95, 'REPRODUCTION', fontsize=12, fontweight='bold')
    plt.figtext(0.025, 0.90, f'GENERATION: {generacion}')
    plt.figtext(0.025, 0.85, f'TIME: {tiempo}')
    plt.figtext(0.025, 0.80, f'Predators: {len(depredadores)}')
    plt.figtext(0.025, 0.75, f'Preys: {len(presas)}')
    plt.figtext(0.025, 0.70, f'New predators: {len(depredadoreshijos)}')
    plt.figtext(0.025, 0.65, f'New preys: {len(presashijos)}')
    
    # Guardar imagen
    plt.savefig(f"reproduccion_{generacion}-{tiempo}.png", dpi=100)
    plt.close(fig)
    
def dibujar_grafica(evolucionpoblaciones):
    
    generaciones = [poblacion[0] for poblacion in evolucionpoblaciones]
    presas = [poblacion[1] for poblacion in evolucionpoblaciones]
    #presas_fallecidas = [poblacion[2] for poblacion in evolucionpoblaciones]
    #presas_nacidas = [poblacion[3] for poblacion in evolucionpoblaciones]
    depredadores = [poblacion[4] for poblacion in evolucionpoblaciones]
    #depredadores_fallecidos = [poblacion[5] for poblacion in evolucionpoblaciones]
    #depredadores_nacidos = [poblacion[6] for poblacion in evolucionpoblaciones]

    colores_presas = ['green', 'limegreen', 'mediumseagreen']
    colores_depredadores = ['red', 'darkred', 'firebrick']

    plt.plot(generaciones, presas, label='presas', color=colores_presas[0])
    #plt.plot(generaciones, presas_fallecidas, label='presas Fallecidas', color=colores_presas[1])
    #plt.plot(generaciones, presas_nacidas, label='nuevos presas', color=colores_presas[2])
    plt.plot(generaciones, depredadores, label='depredadores', color=colores_depredadores[0])
    #plt.plot(generaciones, depredadores_fallecidos, label='depredadores Fallecidos', color=colores_depredadores[1])
    #plt.plot(generaciones, depredadores_nacidos, label='nuevos depredadores', color=colores_depredadores[2])
    plt.xlabel('Generaciones')
    plt.ylabel('Poblaciones')
    # Ajustar la posición y tamaño de la leyenda fuera de la gráfica
    #plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='x-small')
    plt.show()

def graficarevolucion(resumengenD, resumengenP, etiqueta):

    # depredadores
    keneracionesD = []
    depre = {}
    for D, depredador in enumerate(resumengenD):
        keneracionD = depredador['generacion']  # generacion
        identif = depredador['identificador']  # identificador
        valor = depredador[etiqueta]  # valor del parametro ordenada
        keneracionesD.append(keneracionD)  # guardo esos 3 valores en keneraciones
        # en depre se guardaran todos los pares generacion-valordeordenada que tengan el mismo identificador
        if identif not in depre:
            depre[identif] = {'generaciones': [], 'valores': []}
        depre[identif]['generaciones'].append(keneracionD)
        depre[identif]['valores'].append(valor)

    # presas
    keneracionesP = []
    pres = {}
    for P, presa in enumerate(resumengenP):
        keneracionP = presa['generacion']  # generacion
        identif = presa['identificador']  # identificador
        valor = presa[etiqueta]  # valor del parametro ordenada
        keneracionesP.append(keneracionP)  # guardo esos 3 valores en keneraciones
        # en depre se guardaran todos los pares generacion-valordeordenada que tengan el mismo identificador
        if identif not in pres:
            pres[identif] = {'generaciones': [], 'valores': []}
        pres[identif]['generaciones'].append(keneracionP)
        pres[identif]['valores'].append(valor)

    # Configurar colores y marcas
    depredador_color = iter(plt.cm.Reds(np.linspace(0, 1, len(depre))))
    presa_color = iter(plt.cm.Greens(np.linspace(0, 1, len(pres))))
    markers = ['o', 'x']

    # para cada depredador
    for D, (depr, data) in enumerate(depre.items()):
        plt.plot(data['generaciones'], data['valores'], color=next(depredador_color), marker=markers[D % len(markers)], linestyle='-')

    # para cada presa
    for P, (pre, data) in enumerate(pres.items()):
        plt.plot(data['generaciones'], data['valores'], color=next(presa_color), marker=markers[P % len(markers)], linestyle='solid')

    plt.xlabel('Generation')
    plt.ylabel(etiqueta.capitalize())
    plt.grid(True)
    plt.show()
    
# Función para graficar
def graficar_generacion(generaciones, datos, etiqueta_x, etiqueta_y, titulo):
    plt.plot(generaciones, datos, marker='o')
    plt.xlabel(etiqueta_x)
    plt.ylabel(etiqueta_y)
    plt.title(titulo)
    plt.grid(True)
    plt.show()

def animacion():
    # Lista de archivos PNG en la carpeta actual  
    archivos_png = sorted([archivo for archivo in os.listdir() if archivo.endswith('.png')])

    imagenes = [imageio.imread(archivo) for archivo in archivos_png]

    # Guardar la animación en formato GIF    
    imageio.mimsave('animacion.gif', imagenes, duration=0.2)

    return 'animacion.gif'

# VALORES INICIALES
    
numeroD = 20   # NUMERO DE depredadores
reglatrofica = 2  # regla trofica: num presas para alimentar a un depredador
generaciones = 1111  # NUMERO DE GENERACIONES

# PADRES REPRODUCTORES, presas QUE CALIFICAN LIKES
padresD = 0.2         # PORCIÓN MINIMA DE padresD
edadD = 10

# PADRES presas, PRODUCTOS (LOS QUE SE REPONEN), VOTOS,.
padresP = 0.2              # PORCIÓN MINIMA DE padresP
edadP = 5

tasamutacion = 0.1        # TASA DE MUTACIONES

amistad = 10

# GENERACION (gen_tiempo = gen_depredadores = ratioDP * gen_presas)

ratioDP = 1         # generaciones de presas por cada gen depredadores
gen_tiempo = 1     # VIDA DE LA GENERACION         
dt = 0.01           # paso de tiempo, año, mes o como sea que se mida (dt)

# ATRIBUTOS DINAMICOS DE LOS depredadores, PARTIDOS, EMpresas,...

drmaxD = np.pi        # ROTACION      (radianes por segundo) siempre ataca
velocidadD = 3         # VELOCIDAD     (unidades por segundo)

# ATRIBUTOS DINAMICOS DE LAS presas, VOTOS, START-UPs

drmaxP = np.pi * 2        # ROTACION      (radianes por pasotiempo o segundo o dia o lo que sea)
velocidadP = 4        # VELOCIDAD    (unidades por segundo)

# ENTORNO, ECOSISTEMA, ESTANTERIA, URNA,...

x_min = -2.0        # borde izquierdo
x_max = 2.0         # borde derecho
y_min = -2.0        # borde inferior
y_max = 2.0         # borde superior

# RED NEURONAL DE LOS depredadores, presas,... 

numatributos = 2        # NODOS SENSORES
numprocesadores = 5      # NODOS PROCESADORES
numacciones = 2          # NODOS ACTORES

learnrate = 0.1   # velocidad de aprendizaje

# medir tiempos
tiempoinicial = time.process_time()

# hay que controlar que las poblaciones sean sostenibles
    
numpadresD = int(floor(padresD*numeroD)) # num min padres para que sigan las generaciones
numhijosD = numeroD - numpadresD # num de depredadores que deberian reproducirse en la siguiente generacion
numeroP = numeroD*reglatrofica # num de presas   
numpadresP = int(floor(padresP*numeroP))
numhijosP = numeroP - numpadresP
presashijos = [] # solo para inicializar, tras el primer bucle ya no es necesario
depredadoreshijos = []

# vida de la generacion / medida en segundos (o años) del paso de tiempo

pasostiempo=int(gen_tiempo/dt)

ancho = x_max - x_min
alto = y_max - y_min

evolucionpoblaciones = []
resumengenD = [] # depredadores en cada generacion
resumengenP = [] # presas en cada generacion

folder_path = "resultados30"   # aqui se guardaran las generaciones
ultima_generacion_completada = obtener_ultima_generacion(folder_path) # el numero de generacion mayor guardado
if ultima_generacion_completada > 0:
    depredadores, presas, evolucionpoblaciones, resumengenD, resumengenP = cargar_listas_generacion(ultima_generacion_completada, folder_path)
    print(f"Se reanudará desde la generación {ultima_generacion_completada + 1}")
else:
    depredadores,presas = generacion0 (numeroP)
    
for generacion in range(ultima_generacion_completada + 1, generaciones):
# en realidad la vida de la generacion es x3, pues 1/3 a juegos, 1/3 a caza y 1/3 a reproduccion
    # inicializamos las variables:
    criasinviables = 0   
    presasenfermas = 0   
    devoradasP = []  # presas muertas devoradas 
    cachorrosinviables = 0 # mortalidad infantil
    depredadoresenfermos = 0 # depredadores flojos que mejor se descarten como reproductores
    dep = [] # depredadores muertos de hambre
    utilizadasD = 50 # juegos que el earlystopping definira como los necesarios para aprender
    utilizadasP = 50  # aqui solo se inicializan

    # SON JOVENES, JUEGAN PERO NO SE COMEN NI MATAN UNOS A OTROS, ENSAYAN Y ENTRENAN
    # ACUMULANDO EXPERIENCIA SOBRE LAS ESTRATEGIAS (ANGULO Y VELOCIDAD) QUE FUNCIONAN
    # ESTOS DATOS SERAN PARA ENTRENAR CADA UNO A SU RED NEURONAL: APRENDIZAJE
    # el juego define el conjunto de train: valores de los inputs (16) y outputs (2)
    for tiempo in range(1, pasostiempo, 1): # tiempo de juventud 
        amistadD = amistad
        # COMIENZA EL JUEGO               
        for cachorroD,depredador in enumerate(depredadores):
            if depredadores[cachorroD][40] == None: # propone jugar si no tiene experiencia
                distanciaDP = []
            
                for cachorroP,depredador in enumerate(depredadores):
                    # solo si ambos no tienen experiencia, juegan. los mayores ya tienen su set de train
                    if depredadores[cachorroP][40] == None:
                        if cachorroP == cachorroD:
                            DDP = 5 # si no hacemos esta ficcion, se pierde el indice de cachorroP
                        else:
                        # el cachorro que juega de depredador mide la distancia al cachorro depredador mas proximo para que juegue a presa    
                            DDP = sqrt((depredadores[cachorroD][1]-depredadores[cachorroP][1])**2 + (depredadores[cachorroD][2]-depredadores[cachorroP][2])**2)
                        distanciaDP.append(DDP)
                    
                if distanciaDP:
                    # entre los que puede jugar cachorroD, elegimos el cachorro P más próximo
                    DPmin = np.argmin(np.array(distanciaDP)) # el cachorro mas proximo es depredadores[DPmin]

                    if distanciaDP[DPmin] >= 0.1: # para evitar superpuestos (por casualidad o DDP=5)   
                        # prueban valores de su mordisco, salto,... en prueba-error
                        # en cada juego supondremos que prueba secuencialmente: o muerde o salta o ruge, pero no a la vez    
                        # AD es un vector con los valores compartidos por los roles de presa y depredador
                        AD = list(np.intersect1d(depredadores[cachorroD][13], depredadores[DPmin][13]))
                        lenad = len(AD) # si solo tiene un valor sera AD[0], si tiene mas, el prioritario sera AD[0]
                        lenD = len(depredadores[cachorroD][13]) # numatributos en 13
                        lenP = len(depredadores[DPmin][13]) # numatributos en 13
                        triunfos = 0
                        fracasos = 0 # se refiere solo a las estrategias en estos juegos
                        NDP = distanciaDP[DPmin] # inicializa para el primer bucle: la primera experiencia supone que la anterior ha sido satisfactoria
                    
                        # angulo de ataque
                        alfaD = radians(atan2((depredadores[DPmin][2]-depredadores[cachorroD][2]), (depredadores[DPmin][1]-depredadores[cachorroD][1]))) - (depredadores[DPmin][3] - depredadores[cachorroD][3])
                        
                        if amistadD < 1:    # regula la cantidad de juegos para que sea un numero que permita aprender multiplicando o limitando los juegos entre dos amigos
                            amistadD = 1
                            
                        # son mas amigos cuantos mas intereses en comun tengan
                        # con pocos atributos es poco probable que se hagan amigos y menos que sean muy amigos, pero segun crecen los atributos tendran mas experiencias train
                        if lenad != 0: # si ambos no van del mismo rollo en algo, no hay experiencia util y no juegan
                            # es mas facil tener exito por haber aumentado la distancia que acercarse lo suficiente, por lo que los depredadores deberan jugar mas que las presas
                            for _ in range(1,amistadD*lenad): # ahora que se han hecho amigos pues comparten intereses, jugaran un numero de juegos = amistad
                                if len(depredadores[cachorroD][18]) < utilizadasD: # a partir de un numero "utilizados" juegos, maduran y ya se consideran adultos         
                                    ADusado=[]
                                    for i in range(23,39):                                    
                                        # si ambos tienen uno o mas intereses en comun, i.e. uno ofrece certificado de trazabilidad de su producto,
                                        # y el otro busca productos con trazabilidad certificada 
                                        if i in AD and i not in ADusado:                               
                                            # los atributos compartidos AD, tendran prioridad para el depredador distinta de la prioridad de la presa:
                                            # la dispersion de su decision sera menor si es mas prioritario (exploran menos si lo tienen mas claro)
                                            # o cuanto menos prioritario, mas abierto a opciones
                                            # definiremos una lista en 18 de atributos en los que uno, i, va a variar segun criterio anterior
                                            # los otros atributos de AD conservan los valores guardados en 16
                                            posicion_AD = depredadores[cachorroD][13].index(i)
                                            prioridadD = (posicion_AD + 1) / lenad + 1
                                            posicion_AP = depredadores[DPmin][13].index(i)
                                            prioridadP = (posicion_AP + 1) / lenad + 1
                                            
                                            # para dar mas importancia por ejemplo al precio que al tamaño de la pantalla, 
                                            # introducimos la variabilidad segun su prioridad en los atributos heredados
                                            VD = np.random.normal(1,prioridadD)
                                            VAD = depredadores[cachorroD][i] * VD # valor del atributo que se oferta
                                            # para controlar que no nos salgamos del intervalo (0,1), de los valores normalizados del atributo
                                            if VAD <= 0:
                                                VAD = 0.01
                                            elif VAD >= 1:
                                                VAD = 0.99
                        
                                            # si cachorroD juega a precio y DPmin a calidad, cada uno tendrá una variabilidad segun su prioridad
                                            # se encontraran con menos probabilidad que si ambos juegan al mismo rollo
                                            # lo mismo que si uno juega a velocidad y el otro juega a angulo (bajar precio o cambiar orientacion del producto)
                                            VP = np.random.normal(1,prioridadP)
                                            VAP = depredadores[DPmin][i] * VP # valor del atributo que se demanda
                                            if VAP <= 0:
                                                VAP = 0.01
                                            elif VAP >= 1:
                                                VAP = 0.99
                                                
                                            ADusado.append(i)  # en el siguiente bucle no se repetira este i                                  
                                            # cada uno tiene asi un valor en el atributo comun, precio, color ofrecido-disponible, o dientes-cuernos    
                                            # supondremos que cuanto mejor dotado esta el atributo, mejor se acercara al objetivo
                                            # mas claro lo tiene y con menos dudas, mas directo y veloz va
                                            # el paso de tiempo es 1, así que la distancia de la carrera es la velocidadD=depredador[4]
                                            
                                            # cuanto mas valor tiene un atributo, mas intensidad en su incursion en el mercado, mas publicidad, mas inversion, mas velocidad en la carrera
                                            # pero el atributo VAD tiene valor<1
                                            carreraD = depredadores[cachorroD][4]*VAD + np.random.normal(0,depredadores[cachorroD][4]*prioridadD)
                                            if carreraD < 0: carreraD = 0
                                            # ambos tienen intenciones distintas (angulo de ataque con cierta variabilidad, que es su estrategia)
                                            # supondremos que cuanto mas importante es la variable esta en orden correlativo en 13
                                            # atributos mas importantes y de mayor valor, estaran mas focalizados (menor variabilidad en el giro)
                                            alphaD = alfaD + np.random.normal(0,(1-VAD)*prioridadD)
                                            while alphaD > np.pi*2:
                                                alphaD = alphaD - np.pi*2 
                                            while alphaD < -np.pi*2:
                                                alphaD = alphaD + np.pi*2
                                            
                                            # al compartir valores, el que juega a presa, toma la misma decision de acercarse
                                            # pero si su prioridad es menor, su incertidumbre sera mayor
                                            carreraP = depredadores[DPmin][4]*VAP + np.random.normal(0,depredadores[DPmin][4]*prioridadP) 
                                            if carreraP < 0: carreraP = 0
                                            # si huye lo hara en la misma direccion que el que ataca  
                                            alphaP = alfaD + np.random.normal(0,(1-VAP)*prioridadP)
                                            while alphaP > np.pi*2:
                                                alphaP = alphaP - np.pi*2 
                                            while alphaP < -np.pi*2:
                                                alphaP = alphaP + np.pi*2
                                            
                                            # enfilada la proxima presa, va a por ella con un margen estocástico
                                            depredadores[cachorroD][1] = depredadores[cachorroD][1] + carreraD*cos(alphaD)
                                            depredadores[cachorroD][2] = depredadores[cachorroD][2] + carreraD*sin(alphaD)
                
                                            # pero la presa embestira (se gira) al depredador o huira
                                            if VAD > VAP: # huye 
                                                depredadores[DPmin][1] = depredadores[DPmin][1] + carreraP*cos(alphaP)
                                                depredadores[DPmin][2] = depredadores[DPmin][2] + carreraP*sin(alphaP)
                                            else: # embiste porque se gira
                                                depredadores[DPmin][1] = depredadores[DPmin][1] - carreraP*cos(alphaP)
                                                depredadores[DPmin][2] = depredadores[DPmin][2] - carreraP*sin(alphaP)
                                            
                                            # volvemos a calcular la distancia tras la experiencia de ambos
                                            NDPf = sqrt((depredadores[cachorroD][1]-depredadores[DPmin][1])**2 + (depredadores[cachorroD][2]-depredadores[DPmin][2])**2)
                                            
                                            if abs(NDP - NDPf) <= 0.1: # se han acercado
                                                if VAD >= VAP: # precio o calidad ofrecida mejor de la demandada
                                                    # train input
                                                    vectorVAD = depredadores[cachorroD][16]
                                                    vectorVAD[i-23] = VAD
                                                    # desconozco el motivo, pero para evitar un error
                                                    vectorVAD = [0 if val == 0.0 or val == [0.0] else val for val in vectorVAD]                        
                                                    depredadores[cachorroD][18].append(vectorVAD)
                                                    # train output
                                                    depredadores[cachorroD][12].append([carreraD,alphaD])
                                            else:
                                                fracasos += 1
                                            # ahora tienen una nueva distancia entre ellos, sea mayor o menor    
                                            NDP = NDPf
                                
                            depredadores[cachorroD][5].append([triunfos,fracasos])
                            # en 16 estan los valores de los atributos, en 18 estan los valores de exploracion
                            # en 18 tienen cierta variabilidad segun las prioridades y son para explorar y adquirir experiencia
                                    
                            # Aplicar control de frontera
                            controlfronteraD(depredadores[cachorroD])
                            controlfronteraD(depredadores[DPmin])
                            
        # ahora las presas hacen lo mismo
        for criaP,presa in enumerate(presas):
            if presas[criaP][40] == None:
                distanciaPD=[]     
                for criaD,presa in enumerate(presas):                     
                    if presas[criaD][40] == None: # ambos deben ser crias sin experiencia
                        if criaD == criaP:# si se elimina la distancia consigo misma el indice corre una posicion
                            DPD=5
                        # la cria que juega de depredador busca a la cria presa mas proxima y le cita   
                        else:
                            DPD= sqrt((presas[criaP][1]-presas[criaD][1])**2 + (presas[criaP][2]-presas[criaD][2])**2)
                        distanciaPD.append(DPD)

                if distanciaPD:
                    PDmin = np.argmin(np.array(distanciaPD)) # indice de la cria mas proxima a cada criaP           

                    if distanciaPD[PDmin] >= 0.1:    # evitar experiencias de persecucion entre superpuestos               
                        AP =list(np.intersect1d(presas[criaP][13], presas[PDmin][13]))
                        lenap = len(AP) # si solo tiene un valor sera AP[0], si tiene mas, el prioritario sera AD[0]
                        lenP = len(presas[criaP][13]) # numatributos en 13
                        lenD = len(presas[PDmin][13]) # numatributos en 13
                        triunfos = 0  # de la criaP
                        fracasos = 0  # de la criaP
                        NPD = distanciaPD[PDmin] # solamente sirve para inicializar
                        # angulo de huida 
                        alfaP = radians(atan2((presas[criaP][2]-presas[PDmin][2]), (presas[criaP][1]-presas[PDmin][1]))) - (presas[criaP][3] - presas[PDmin][3])
                        # angulo de persecucion
                        alfaD = - alfaP
                        # las crias pueden ser mas y para compensar que aprendan mas que los depredadores... 
                        amistadP = int(amistad/reglatrofica)
                        if amistadP < 1: amistadP = 1
                        if lenap != 0: # si ambos no van del mismo rollo, no hay experiencia util y no juegan                        
                            for _ in range(1,amistadP * lenap): # pero si van de lo mismo, se hacen amiguetes y juegan muchas veces
                                if len(presas[criaP][18]) < utilizadasP:  # limitamos los juegos a un valor suficiente para aprender
                                    APusado=[]
                                    for i in range(23,39):
                                        if i in AP and i not in APusado:
                                            posicion_AP = presas[criaP][13].index(i)
                                            prioridadP = (posicion_AP + 1) / lenap + 1
                                            posicion_AD = presas[PDmin][13].index(i)
                                            prioridadD = (posicion_AD + 1) / lenap + 1
                                            
                                            # para dar mas importancia por ejemplo a los cuernos que a las pezuñas, 
                                            # introducimos la variabilidad segun su prioridad en los atributos heredados
                                            VP = np.random.normal(1,prioridadP)
                                            VAP = presas[criaP][i] * VP # valor del atributo que se oferta
                                            # para controlar que no nos salgamos del intervalo (0,1), de los valores normalizados del atributo
                                            if VAP <= 0:
                                                VAP = 0.01
                                            elif VAP >= 1:
                                                VAP = 0.99
                        
                                            # si criaP juega a empujar y PDmin a morder, cada uno tendrá una variabilidad segun su prioridad
                                            # se encontraran con menos probabilidad que si ambos juegan al mismo rollo
                                            VD = np.random.normal(1,prioridadD)
                                            VAD = presas[PDmin][i] * VD # valor del atributo que se demanda
                                            if VAD <= 0:
                                                VAD = 0.01
                                            elif VAD >= 1:
                                                VAD = 0.99
                                                
                                            # criaP juega a huir, lo que ya esta considerado en alfaP
                                            carreraP = presas[criaP][4]*VAP + np.random.normal(0,presas[criaP][4]*prioridadP)
                                            if carreraP < 0: carreraP = 0
                                            alphaP = alfaP + np.random.normal(0,(1-VAP)*prioridadP)
                                            while alphaP > np.pi * 2:
                                                alphaP = alphaP - np.pi * 2 
                                            while alphaP < -np.pi * 2:
                                                alphaP = alphaP + np.pi * 2
                                            
                                            carreraD = presas[PDmin][4]*VAD + np.random.normal(0,presas[PDmin][4]*prioridadD)
                                            if carreraD < 0: carreraD = 0
                                            alphaD = alfaD + np.random.normal(0,(1-VAD)*prioridadD)
                                            while alphaD > np.pi * 2:
                                                alphaD = alphaD - np.pi * 2 
                                            while alphaD < -np.pi * 2:
                                                alphaD = alphaD + np.pi * 2
                                            
                                            presas[criaP][1] = presas[criaP][1] + carreraD * cos(radians(alphaP))
                                            presas[criaP][2] = presas[criaP][2] + carreraD * sin(radians(alphaP))
                                            
                                            if VAP > VAD: # huir  
                                                presas[PDmin][1]= presas[PDmin][1] + carreraP * cos(alphaD)
                                                presas[PDmin][2]= presas[PDmin][2] + carreraP * sin(alphaD)
                                            else: # embestir
                                                presas[PDmin][1]= presas[PDmin][1] - carreraP * cos(alphaD)
                                                presas[PDmin][2]= presas[PDmin][2] - carreraP * sin(alphaD)
        
                                            # volvemos a calcular la distancia tras la experiencia de ambos
                                            NPDf = sqrt((presas[criaP][1]-presas[PDmin][1])**2 + (presas[criaP][2]-presas[PDmin][2])**2)
                                           
                                            if abs(NPD - NPDf) >= 0.1: # si ha conseguido huir, sea cual sea el tamaño de sus cuernos o 
                                                if VAP >= VAD: # precio o calidad ofrecida mejor de la demandada
                                                    # train input
                                                    vectorVAP = presas[criaP][16]
                                                    vectorVAP[i-23] = VAP
                                                    vectorVAP = [0 if val == 0.0 or val == [0.0] else val for val in vectorVAP]
                                                    presas[criaP][18].append(vectorVAP)
                                                    # train output
                                                    presas[criaP][12].append([carreraP,alphaP])
                                                    triunfos += 1
                                                else:
                                                    fracasos += 1
                                            else: fracasos +=1
                                            
                                            NPD = NPDf 
                                        
                            presas[criaP][5].append([triunfos,fracasos])
        
                # Aplicar control de frontera
                controlfronteraP(presas[criaP])
                controlfronteraP(presas[PDmin])
    
    # la mortalidad infantil es necesaria para el programa (individuos que tienen una configuracion de atributos
    # tal que no podran aprender), pero tambien en la realidad se descartan los pollitos que menos pian o en el futbol quien no va al entreno
    if generacion < 10: juegosD = generacion
    else: juegosD = 10

    # Depurar la lista que contiene solo los depredadores que han acumulado experiencia (viejos y nuevos jugones)  
    apaticosD = [depredador for depredador in depredadores if len(depredador[18]) <= juegosD]
    # ahora los ordenamos para que se eliminen primero los de la generacion anterior que pudieran quedar
    apaticosD = sorted(apaticosD, key=lambda depredador: depredador[0])

    for depredador in depredadores[:]:
        if depredador in apaticosD:
            cachorrosinviables += 1
            depredadores.remove(depredador)
            # precaucion por si hay demasiados cachorros sin experiencia
            if len(depredadores) <= numpadresD*2: break
            # los cachorros sin experiencia no eliminados tendran una segunda oportunidad de adquirir train en la proxima generacion
        else: # ya que estamos depuro un error extraño que asigna [0.0] a algunos sitios
            depredador[16] = [0 if val == 0.0 or val == [0.0] else val for val in depredador[16]]
            depredador[18] = [0 if val == 0.0 or val == [0.0] else val for val in depredador[18]]
    print('quedan', len(depredadores),'depredadores para iniciar caza, de los que', len(apaticosD)- cachorrosinviables, 'no son activos')
            
    valoresD = [len(sublistaD[18]) for sublistaD in depredadores if sublistaD[18]]
    if valoresD:
        mediaD = sum(valoresD) / len(valoresD)
        #print("La media de las experiencias en depredadores:", mediaD)
    else:
        print("No hay valores para calcular la media de experiencias para los cachorros")
    
    # Como a las presas les cuesta comenzar a jugar, les damos un par de generaciones de ventaja
    if generacion < 10: juegosP = generacion
    else: juegosP = 10 
    
    # Construir apaticosP dentro del bucle donde se eliminan las presas inútiles
    apaticosP = [presa for presa in presas if len(presa[18]) <= juegosP]
    # en el caso en que haya demasiados inutiles, no nos los podemos cargar a todos 
    apaticosP = sorted(apaticosP, key=lambda presa: presa[0])

    for presa in presas[:]:
        if presa in apaticosP:
            criasinviables += 1
            presas.remove(presa)
            if len(apaticosP) <= numpadresP*2: break
        else: # ya que estamos depuro un error extraño que asigna [0.0] a algunos sitios
            presa[16] = [0 if val == 0.0 or val == [0.0] else val for val in presa[16]]
            presa[18] = [0 if val == 0.0 or val == [0.0] else val for val in presa[18]]
    print('quedan', len(presas), 'presas para ser cazadas, de las que', len(apaticosP)-criasinviables, 'estan atontadas')
    
    valoresP = [len(sublistaP[18]) for sublistaP in presas if sublistaP[18]]
    if valoresP:
        mediaP = sum(valoresP) / len(valoresP)
        #print("La media de las experiencias en presas:", mediaP)
    else:
        print("No hay valores para calcular la media de experiencias para las crias")
    
    # LOS CACHORROS SUPERVIVIENTES DESARROLLAN SUS REDES NEURONALES CON LAS EXPERIENCIAS DEL JUEGO       
    # deben prever cuales de las variabilidades sobre angulos y distancias son mas exitosas
    for depredador in depredadores[:]:
        if depredador[40] == None and len(depredador[18]) > 10: # esto selecciona a los nuevos con experiencia
            depredador, tiemporeaccionD, utilizadasD = neuralD(depredador)
            depredador[14].append(utilizadasD)
            # tiempo para realizar el aprendizaje en cada juego
            depredador[39].append(tiemporeaccionD)
            #print('cachorro', depredador[0] ,'aprende de', len(depredador[18]) ,'experiencias')
        else: # reubicamos a todos los que van a seguir en el ecosistema
            depredador[1] = np.random.uniform(x_min,x_max)
            depredador[2] = np.random.uniform(y_min,y_max) 
            
    # LAS CRIAS SUPERVIVIENTES DESARROLLAN SUS REDES NEURONALES CON LAS EXPERIENCIAS DEL JUEGO
    # la distribucion de estrategias ha sido gaussiana, pero su seleccion de exitos no
    # segun los atributos, numprocesadores y experiencias, cada una tendrá una red distinta
    for presa in presas[:]:
        if presa[40] == None and len(presa[18]) > 10: # solo calculamos la red para los jovenes que juegan
            presa, tiemporeaccionP, utilizadasP = neuralP(presa)
            presa[14].append(utilizadasP)
            # tiempo para realizar el aprendizaje
            presa[39].append(tiemporeaccionP)
            #print('cria', presa[0] ,' aprende de', len(presa[18]),'experiencias')
        else: # controlaremos que no se hayan escapado demasiado lejos
            presa[1] = np.random.uniform(x_min,x_max)
            presa[2] = np.random.uniform(y_min,y_max)        
    
    # COMIENZA LA CAZA DONDE SE SELECCIONARAN LAS MEJORES DECISIONES (conjunto de test)  
    # los bichos ya son redes neuronales individuales y distintas a partir de similares experiencias
    # pero los distintos atributos, configuraciones de la red y eficiencia en tiempo, las hace diversas en opciones de supervivencia      
    print('comienza la caza')
    carrona = 0
    cazza = 0
    nocaza = 0
    persecucion = 0
    # inicializacion por si en los primeros momentos o por falta de buenos depredadores o presas, no se pasara el bucle de caza
    DPcarrona = 0 # recuento por cada depredador
    DPcazza = 0
    DPnocaza = 0
    DPpersecucion = 0
    for tiempo in range(1, pasostiempo, 1):
        if len(presas) >= numpadresP + 2:# Verificamos si hay suficientes presas para continuar con el proceso de caza
            for D, depredador in enumerate(depredadores):            
                DPcarrona = 0 # recuento por cada depredador
                DPcazza = 0
                DPnocaza = 0
                DPpersecucion = 0
                if depredador[40] != None: 
                    distanciaDP = []  # Lista para las distancias de cada depredador a todas las presas
                    for P, presa in enumerate(presas):
                        DPP = sqrt((depredador[1] - presa[1])**2 + (depredador[2] - presa[2])**2)
                        distanciaDP.append(DPP)
                    
                    if distanciaDP: # para controlar generaciones sin eventos de caza
                        DPmin = np.argmin(np.array(distanciaDP))  # Índice de la presa más cercana
                        #print('depredador', depredador[0], 'intenta cazar a', presas[DPmin][0])
                        
                        # depredador HA ALCANZADO A UNA presa: COME FITNESS... si puede
                        if presas[DPmin][40] == None:     # si no ha aprendido es apaticaP que no ha sido descartada
                            carrona += 1
                            DPcarrona += 1
                            devoradasP.append(presas[DPmin][0])  # presas comidas en cada unidad de tiempo por todos los depredadores
                            depredador[8] += presa[8] / (2*reglatrofica)  # aprovecha menos la energía trofica que si se lo curra
                            presas.remove(presas[DPmin])
                            #print(depredador[0], 'carroña')
                            
                        elif distanciaDP[DPmin] <= 0.1:  # SI NO, TRAS LA PERSECUCION PUEDE HABERLA ALCANZADO
                            # si la presa es mejor en algun atributo que el depredador, se libra
                            # por ejemplo la presa grita mas de lo que ruge el depredador
                            # esto se puede detallar mucho, pero para el objetivo, no importa
                            # i.e. un ataque mas o menos frontal contra una presa grande y con cuernos grandes, 
                            # provoca la muerte del depredador si ademas es poco agresivo y poco ágil, 
                            # pero si es ágil libra sin comer y si no es suficientemente agresivo, pierde peso...
                            # reglas a implementar que no aportan demasiado, eso puede implicar premios y castigos en sus atributos
                            
                            # inicializamos tiempos unitarios en 39 con datos del 22 (en juegos)
                            # con la experiencia en atacar y huir, se van tomando decisiones
                            # si uno de los dos duda y tarda más en decidir: pierde
                            if depredador[39]:
                                depredador[22] = st.mean(depredador[39])
                            if presas[DPmin][39]:
                                presas[DPmin][22] = st.mean(presas[DPmin][39])
                            
                            depredador, presas[DPmin], exito = caza(depredador, presas[DPmin])
                            
                            if exito == True:
                                devoradasP.append(presas[DPmin][0])  # presas comidas en cada unidad de tiempo por todos los depredadores
                                presas[DPmin][40] = None   # otro depredador puede carroñear y si no se recuperara en la siguiente generacion
                                # esto es para evitar que el bucle incluya un break al comprar y siga comprando
                                cazza += 1
                                DPcazza += 1
                                #print(depredador[0],'caza')
                            else: # se va a otro lado
                                nocaza += 1
                                DPnocaza += 1
                                presas[DPmin][1] = np.random.uniform(x_min,x_max)
                                presas[DPmin][2] = np.random.uniform(y_min,y_max)
                                #print(depredador[0], 'fracasa')
                                presas[DPmin][6].append(DPnocaza)
                            
                            controlfronteraD(depredador)
                            controlfronteraP(presas[DPmin])
                                
                        # equivale al set de test tanto para presas como para depredadores:
                        else:    # depredador NO HA ALCANZADO A ALGUNA presa: PERSIGUE
                            persecucion += 1
                            DPpersecucion += 1
                            # Llamar a la función decisionD con la información del depredador
                            alphaD, carreraD, tiempoD = decisionD(depredador)
                            
                            depredador[39].append(tiempoD) # guardamos la eficiencia de la decision
                            depredador[10].append([carreraD,alphaD])
                            # enfilada la proxima presa, va a por ella con un margen estocástico
                            depredador[1] = depredador[1] + carreraD * cos(radians(alphaD))
                            depredador[2] = depredador[2] + carreraD * sin(radians(alphaD))
                            # la presa escapa pero va mas lenta
                            # corre en sentido contrario también con variabilidad (se le suma cuando los 180º)
                            
                            # Llamar a la función decisionP con la información de la presa
                            alphaP, carreraP, tiempoP = decisionP(presas[DPmin])
                            
                            presas[DPmin][39].append(tiempoP) # guardamos la eficiencia de la decision 
                            presas[DPmin][10].append([carreraP,alphaP])
                            
                            # el elefante ataca al leon
                            # vamos a provocar que la proactividad del depredador se premie
                            if presas[DPmin][8] > depredador[8]: # embiste 
                                presas[DPmin][1]= presas[DPmin][1] + carreraP * cos(alphaP)
                                presas[DPmin][2]= presas[DPmin][2] + carreraP * sin(alphaP)
                                depredador[8] -= 1 / pasostiempo # gasta energía en perseguir
                                presas[DPmin][8] += reglatrofica / pasostiempo # gasta energía en huir o embestir

                            else: # huye
                                presas[DPmin][1]= presas[DPmin][1] - carreraP * cos(alphaP)
                                presas[DPmin][2]= presas[DPmin][2] - carreraP * sin(alphaP)
                                depredador[8] += 1 / pasostiempo 
                                presas[DPmin][8] -= reglatrofica / pasostiempo # gasta energía en huir o embestir

                            controlfronteraD(depredador)
                            controlfronteraP(presas[DPmin])
                        
                        depredador[6].append(DPcazza)
                        # Aplicar control de frontera
                    
                #print ('depredador',depredador[0],'ha perseguido', persecucion, 'ha carroñeado',carrona, 'ha cazado', cazza, 'pero se le han escapado',nocaza)
        
        for P,presa in enumerate(presas):
            # las plantas fotosintetizan, los herbívoros comen hierba, solo con el tiempo ganan alimentacion
            # gastan energia en comer, huir, atacar,... por eso suponemos que comen el doble de lo que como minimo van a gastar en vivir
            # tiene que haber de mas para invertirlo en los hijos
            presa[8] += 2*reglatrofica / pasostiempo
            # variable que mejora con la experiencia durante la temporada (baja la necesidad)
            if presa [23] != 0:    
                presa[23] += 1 / (pasostiempo*reglatrofica) # deseo un vino o una antiguedad, una marca (argumento de fundado en 1918)
            # variable que empeora con la experiencia (aumenta la utilidad a medida que pasa la temporada)
            if presa [24] != 0:    
                presa[24] -= 1 / (pasostiempo*reglatrofica)   # perecederos, verduras, inmuebles, coches,...  
            # variable que mejora con el ejercicio o con dedicarse mas a los escaparates
            if presa [25] != 0 and persecucion != 0:    
                presa[25] += DPpersecucion / persecucion  # en el orden de 100 persecuciones y 100 (pasostiempo*reglatrofica), hay que normalizar
            # variable que empeora con el ejercicio, de tanto mirar nos liamos sobre lo que queremos
            if presa [26] != 0 and persecucion != 0:    
                presa[26] -= DPpersecucion / persecucion   # cansancio y al final compras cualquier cosa
            # variable que mejora con la exigencia del presa
            if presa [27] != 0 and DPnocaza != 0:    
                presa[27] += DPnocaza / nocaza   # al alcance de unos pocos, o lujo  
            # variable que empeora con la exigencia del presa
            if presa [28] != 0 and DPnocaza != 0:    
                presa[28] -= DPnocaza / nocaza  # en el orden de 150
            # variable que mejora con la alimentacion, cuanto más compro más quiero comprar
            if presa [29] != 0:    
                presa[29] += presa[8]/(pasostiempo*reglatrofica)  # un luchador de sumo, o una morsa
            # variable que empeora con la alimentacion, el carro se llena y no puedo cargar con mas bolsas
            if presa [30] != 0:    
                presa[30] -= presa[8]/(pasostiempo*reglatrofica)  # sobrepeso y opciones de ligar
            # variable que mejora con la edad y la alimentacion
            if presa [31] != 0:    
                presa[31] += presa[8]*presa[9]/(pasostiempo*reglatrofica)  # los colmillos de un elefante, 
            # variable que empeora con la edad y la alimentacion
            if presa [32] != 0:    
                presa[32] -= presa[8]*presa[9]/(pasostiempo*reglatrofica)  # quiero un programa de software probado, sin buggs
            # variable que mejora con la edad, pero engordar no favorece
            if presa [33] != 0 and presa[8] != 0:    
                presa[33] += presa[9]/(presa[8]*(pasostiempo*reglatrofica)) # quiero seguridad de un producto maduro, con prestigio, estable
            # variable que empeora con la edad pero se aprecia la hermosura de las canas
            if presa [34] != 0 and presa[8] != 0:    
                presa[34] -= presa[9]/(presa[8]*(pasostiempo*reglatrofica)) # el gorila que por gordo pierde su agilidad
            # variable que mejora con los hijos
            if presa [35] != 0 and presa[7] != 0:    
                presa[35] += presa[9]/presa[7] # cuanto mas sales de marcha, mas te conoces los garitos
            # variable que empeora con los hijos
            if presa [36] != 0 and presa[7] != 0:    
                presa[36] -= presa[9]/presa[7]  # en el orden de las 4 crias por padre
            # variable que mejora con los intereses
            if presa [37] != 0:    
                presa[37] += len(presa[13])/((pasostiempo*reglatrofica)*16) # si gusta el cine y el deporte, mejor que si gustan los videojuegos para salir 
            # variable que empeora con el intereses
            if presa [38] != 0:    
                presa[38] -= len(presa[13])/((pasostiempo*reglatrofica)*16)   # en el orden de 

            # para evitar valores negativos
            for i in range(23,39): 
                if presa[i] < 0:
                    presa[i] = 0
            # actualizacion del vector de valores de los atributos, en 16
            for i in range(0,len(presa[13])): # len13 <= len16 ==16
                 presa[16][presa[13][i]-23]= presa[presa[13][i]]

        # AHORA DERPREDADORES
        # recuento en dep de los que moriran de hambre tras pasar esta unidad de tiempo
        for D,depredador in enumerate(depredadores):
            # variable que mejora con la experiencia
            if depredador [23] != 0:    
                depredador[23] += 1 / pasostiempo # ofrezco un vino o una antiguedad, una marca (argumento de fundado en 1918)
            # variable que empeora con la experiencia
            if depredador [24] != 0:    
                depredador[24] -= 1 / pasostiempo   # perecederos, verduras, inmuebles, coches,...  
            # variable que mejora con el ejercicio o con la venta agresiva
            if depredador [25] != 0 and DPpersecucion != 0:    
                depredador[25] += DPpersecucion / persecucion  # en el orden de 100 persecuciones y 100 pasostiempo, hay que normalizar
            # variable que empeora con el ejercicio, dudando no se cierra el trato
            if depredador [26] != 0 and DPpersecucion != 0:    
                depredador[26] -= DPpersecucion / persecucion   # cansancio y al final compras cualquier cosa
            # variable que mejora con el prestigio
            if depredador [27] != 0 and DPcazza != 0 :    
                depredador[27] += DPcazza / cazza  # al alcance de unos pocos, o lujo  
            # variable que empeora con el exito
            if depredador [28] != 0 and DPcazza != 0:    
                depredador[28] -= DPcazza / cazza  # en el orden de 80
            # variable que mejora con la alimentacion
            if depredador [29] != 0:    
                depredador[29] += depredador[8] / pasostiempo  # un luchador de sumo, o una morsa
            # variable que empeora con la alimentacion
            if depredador [30] != 0:    
                depredador[30] -= depredador[8] / pasostiempo  # sobrepeso y opciones de ligar
            # variable que mejora con la edad y la alimentacion
            if depredador [31] != 0:    
                depredador[31] += (depredador[8]*depredador[9]) / pasostiempo  # los colmillos de un tigre 
            # variable que empeora con la edad y la alimentacion
            if depredador [32] != 0:    
                depredador[32] -= (depredador[8]*depredador[9]) / pasostiempo  # ofrezco un programa de software probado, sin buggs
            # variable que mejora con la edad, pero engordar no favorece
            if depredador [33] != 0 and depredador[8] != 0:    
                depredador[33] += depredador[9] / (depredador[8]*pasostiempo) # ofrezco seguridad de un producto maduro, con prestigio, estable
            # variable que empeora con la edad pero se aprecia la hermosura de las canas
            if depredador [34] != 0 and depredador[8] != 0:    
                depredador[34] -= depredador[9] / (depredador[8]*pasostiempo) # el leon que por gordo pierde su agilidad
            # variable que mejora con la edad
            if depredador [35] != 0 and depredador[7] != 0:    
                depredador[35] += depredador[9] / depredador[7] # en el orden de los 10 cachorros por padre
            # variable que empeora con la edad
            if depredador [36] != 0 and depredador[7] != 0:    
                depredador[36] -= depredador[9] / depredador[7]  # con hijos del anterior matrimonio, se liga menos
            # variable que mejora con los intereses
            if depredador [37] != 0:    
                depredador[37] += len(depredador[13])/(pasostiempo*16) # amplio catalogo 
            # variable que empeora con el intereses
            if depredador [38] != 0:    
                depredador[38] -= len(depredador[13])/(pasostiempo*16)   # coñazo o petardo, cansino 
            
            # para que no haya valores negativos
            for i in range(23,39):
                if depredador[i] < 0:
                    depredador[i] = 0
            # actualizacion del vector de valores de los atributos, en 16
            for i in range(0,len(depredador[13])):
                 depredador[16][depredador[13][i]-23]= depredador[depredador[13][i]]
            
            # aguantan para reproducirse incluso con hambre para evitar extinción
            if depredador[8] <= 0 and len(depredadores) <= numhijosD + 2: # +2 para evitar que los gemelos se lo salten
                dep.append(depredador[0]) # se muere de hambre
                depredadores.remove(depredador)
                # nunca habra menos de numpadresD +1
        
        # certificamos la muerte de las devoradas por cualquier depredador   
        presas = [presa for presa in presas if presa not in devoradasP]
        # certificamos la defuncion de los que no cazan lo suficiente o han enfermado
        depredadores = [depredador for depredador in depredadores if depredador not in dep]    
        
        # los viejos van a dejar paso a las nuevas generaciones, aunque sean exitosos y sirven de carroña
        for P, presa in enumerate(presas):
            if presa[9] > edadP:
                presa[40] = None
                
        for D, depredador in enumerate(depredadores):
            if depredador[9] > edadD:
                presa[40] = None
            
        # GRAFICO (llama a la funcion dibujar) PARA CADA PASO DE CADA GENERACION 
        #dibujarcaza(depredadores,presas,generacion,tiempo)    # devuelve grafico png
    
    print('carroña', carrona, 'caza', cazza, 'escapan', nocaza, 'persecuciones', persecucion)
    print('antes de evaluarlos, quedan', len(depredadores), 'depredadores y', len(presas), 'presas')
    
    # YA HAN COMIDO Y SE HAN CAZADO HASTA QUEDAR len(depredadores) y len(presas),
    # TOCA REPRODUCIRSE Y RESTITUIR EL NUMERO DE criasinviables + devoradasP con numhijosP, 
    # LOS MEJORES REPRODUCTORES HAN SOBREVIVIDO, COMO MIN numpadresD y numpadresP   
    # puede suceder que en algunas generaciones no haya renovacion suficiente,
    # el programa sugiere la necesidad de que viejos y mal alimentados perezcan de enfermedad
    # si no podriamos estancarnos en mesetas poblacionales 
    while len(depredadores) >= numpadresD + 2:
        valor_minimoD = min(depredador[8] for depredador in depredadores)
        if valor_minimoD <= 1: # hasta suficientemente alimentados
            depredadormin = next(depredador for depredador in depredadores if depredador[8] == valor_minimoD)
            dep.append(depredadormin[0])
            depredadoresenfermos += 1
            depredadores.remove(depredadormin)
            #print('depredador', depredadormin[0] ,'se muere de hambre')
        else: break
    
    while len(presas) >= numpadresP + (2*reglatrofica):
        presas_activas = [presa for presa in presas if presa[40] is not None]  # Filtrar las presas activas
        valor_minimoP = min(presa[8] for presa in presas_activas)   
        if valor_minimoP <= reglatrofica:
            # Filtrar la presa que tiene el valor mínimo de alimentación, y pasan a ser inactivas
            presamin = next(presa for presa in presas_activas if presa[8] == valor_minimoP)
            presamin[40] = None
            presasenfermas += 1
            #print('presa', presamin[0] ,'se muere de hambre')
        else: break
            
    print('han enfermado', depredadoresenfermos, 'depredadores y', presasenfermas, 'presas')
    
    alimentacionP = [presa[8] for presa in presas]
    MalimentacionP = st.mean(alimentacionP)
    alimentacionD = [depredador[8] for depredador in depredadores]
    MalimentacionD = st.mean(alimentacionD)
   
    print('alimentacion de los depredadores', MalimentacionD, 'alimentacion de las presas', MalimentacionP)
    
    # presas CORTEJAN CON presas Y depredadores CON depredadores
    #print('comienza la fase reproductiva')
    # presa BUSCA presa (seguimos en la misma generacion, pero entran en celo)
    SVP = len(presas)   # presas supervivientes, mayor o igual que numpadresP
    presashijos = []           
    numpresashijos = 0
    identificadorD = numeroD*(generacion+1)
    
    # depredador BUSCA depredador
    SVD = len(depredadores)   # depredadores supervivientes, mayor o igual que numpadresP
    depredadoreshijos = []
    numdepredadoreshijos = 0
    identificadorP = numeroP*(generacion+1)    # para numerar a las presas
    
    # puede que haya mas necesidad de reproduccion que la que es capaz de suministrar en el tiempo de la generacion
    # por ello pondremos un adaptador de los ciclos reproductivos que dependera de las crias y cachorros que caben
    promiscuidad = int((numeroP+numeroD)/(SVP+SVD))
    for tiempo in range(pasostiempo*promiscuidad):
        # bucle hasta que se haya restablecido el numeroP
        if numpresashijos + SVP <= numeroP:# SVP mayor o igual que numpadresP
            # pasan de comer y buscan novios
            for padreP, presa in enumerate(presas):
                distanciaPP=[]
                for madreP, presa in enumerate(presas):
                    if padreP == madreP:
                        PMP=5 # no se corre el indice de los posteriores
                    else:
                        PMP=sqrt((presas[padreP][1]-presas[madreP][1])**2 + (presas[padreP][2]-presas[madreP][2])**2)
                        # de momento no son selectivos, pues han sobrevivido
                        distanciaPP.append(PMP) 
                        # se selecciona por distancia   
       
                if distanciaPP:
                    PPmin=np.argmin(np.array(distanciaPP)) # novia mas proxima
                
                    if distanciaPP[PPmin] <= 0.1:
                        presas[padreP][7] += 1 # contador de eventos reproductivos
                        presas[PPmin][7] += 1                   
                        numpresashijos += 1
                        # para reproducirse van a gastar energía
                        presas[padreP][8] -= 1 / pasostiempo 
                        presas[PPmin][8] -= 1 / pasostiempo  
                        
                        # presas que se reproducen  
                        atributosP_hijo1 = [] # inicializacion 
                        atributosP_hijo1 = hijo(presas[padreP][13],presas[PPmin][13])
                        # los valores de los inputs no se heredan, si su seleccion
                        
                        # ahora con el numero de procesadores de la capa oculta
                        Np = presas[padreP][17]
                        Mp = presas[PPmin][17]
                        # tamaño más próximo a la media de N y M
                        numprocesadores_hijo1 = int(np.round((Np + Mp) / 2))
    
                        atributos_hijo1 = []
                        exitostrainP = [] # lista vacia de dos valores con un numero variable de entradas
                        exitostestP = []
                        exitostestRP = []
                        tiemposP = [] # lista vacia de un valor con un numero variable de entradas
                      
                        identificadorP += 1
                        presa_hijo1= [identificadorP,
                                     presas[padreP][1], # posicion aleatoria de cada presa
                                     presas[PPmin][2],
                                     np.random.uniform(0,drmaxP), # angulo
                                     np.random.uniform(0,velocidadP), # velocidad
                                     [], # 5, juegos
                                     [], # 6, contador de eventos caza
                                     0, # 7, contador de eventos reproductivos           
                                     1, # 8, fitness 
                                     0, # 9, edad 
                                     exitostestP,
                                     exitostestRP, #11, exitos reproductivos
                                     exitostrainP, # 12, tabla de angulos y velocidades con exito (entrenamiento)
                                     atributosP_hijo1, # 13,lista de atributos seleccionados
                                     [], # 14, contador de entrenamientos
                                     0, # 15, contador de conexiones
                                     # los hijos heredan los valores y las configuraciones de los padres
                                     atributos_hijo1, # lista de los valores
                                     numprocesadores_hijo1,
                                     [],
                                     # los pesos no se heredan (de Lamarck a Darwin)
                                     [], # Inicializacion entre la entrada y la capa oculta (solo hereda el numero, no los pesos)           
                                     [], # Inicializacion dentro de la capa oculta
                                     [],# Inicialización de los pesos de las conexiones entre la capa oculta y la capa de salida
                                     0, # tiempo medio de decision
                                     0, # 23, agilidad
                                     0, # 24, agresividad          
                                     0, # 25, cocea
                                     0, # 26, embiste  
                                     0, # 27, tamaño de sus defensas (cuernos, espinas, veneno, camuflaje,...)
                                     0, # 28, se camufla
                                     0, # 29, apesta o escupe
                                     0, # 30, grita 
                                     0, # 31, atributo de reserva 
                                     0, # 32, atributo de reserva 
                                     0, # 33, atributo de reserva 
                                     0, # 34, atributo de reserva 
                                     0, 
                                     0,
                                     0,
                                     0,
                                     tiemposP,
                                     None
                                     ]
                        
                        # MUTACION
                        atributosP_hijo1, numprocesadores_hijo1 = mutacion (presa_hijo1[13],presa_hijo1[17])
                        presa_hijo1[13] = atributosP_hijo1
                        presa_hijo1[17] = numprocesadores_hijo1
                        #print('la cria ha nacido con', len(presa_hijo1[13]) ,'atributos y', presa_hijo1[17] ,'procesadores')
                            
                        # ACTUALIZACION DE LA presa PARA ENTRAR EN LA SIGUIENTE GENERACION
                        # ponemos a 0 los atributos no seleccionados y actualizamos los campos entre 23 y 38
                        for i in range(23,39):  # Campos entre 23 y 38 (ambos inclusive)
                            if i not in atributosP_hijo1:
                                presa_hijo1[i] = 0
                            else:
                                presa_hijo1[i] = np.random.uniform(0,0.5)
                            # actualizamos tambien 16:
                            atributos_hijo1.append(presa_hijo1[i]) 
                                       
                        presashijos.append(presa_hijo1)

                        if presas[PPmin][8] <= presas[padreP][8] :
                            # si el padre está bien alimentado, puede reproducirse mejor
                            numpresashijos += 1 # HERMANO GEMELO
                            presas[padreP][8] -= 1 / pasostiempo # gasta energia
                            presas[PPmin][8] -= 1 / pasostiempo  # gasta energia
    
                            atributosP_hijo2 = []
                            atributosP_hijo2 = hijo(presas[padreP][13],presas[PPmin][13])
                            
                            # ahora con el numero de procesadores de la capa oculta
                            Np = presas[PPmin][17]
                            Mp = presas[padreP][17]
                            # tamaño más próximo por arriba a la media de N y M                      
                            numprocesadores_hijo2= int(np.round((Np + Mp) / 2))
                            
                            atributos_hijo2=[]
                            exitostrainP=[] # lista vacia de dos valores con un numero variable de entradas
                            exitostestP=[]
                            exitostestRP=[]
                            tiemposP=[] # lista vacia de un valor con un numero variable de entradas
                            # identificar
                            identificadorP += 1
                            presa_hijo2= [identificadorP,
                                         presas[PPmin][1], # posicion aleatoria de cada presa
                                         presas[padreP][2],
                                         np.random.uniform(0,drmaxP), # angulo
                                         np.random.uniform(0,velocidadP), # velocidad
                                         [], # juegos
                                         [], # 6, contador de eventos caza
                                         0, # 7, contador de eventos reproductivos           
                                         1, # 8, fitness al nacer 
                                         0, # 9, edad 
                                         exitostestP, #10, contador de exitos de huida
                                         exitostestRP, # 11, contador de exitos reproductivos
                                         exitostrainP, # 12, tabla de angulos y velocidades con exito
                                         atributosP_hijo2, # 13, tabla de angulos y velocidades que no han conseguido atrapar a la presa (entrenamiento)
                                         [], # 14, contador de entrenamientos
                                         0, # 15, contador de conexiones
                                         # los hijos heredan los valores y las configuraciones de los padres
                                         atributos_hijo2, # valores y 0's
                                         numprocesadores_hijo2,
                                         [],
                                         # los pesos no se heredan (de Lamarck a Darwin)
                                         [], # pesos capa input-hidden
                                         [], # pesos capa recurrente hidden-hidden
                                         [], # pesos capa hidden-output
                                         0, # tiempo de decision
                                         0, # 23, agilidad
                                         0, # 24, agresividad          
                                         0, # 25, cocea
                                         0, # 26, embiste  
                                         0, # 27, tamaño de sus defensas (cuernos, espinas, veneno, camuflaje,...)
                                         0, # 28, se camufla
                                         0, # 29, apesta o escupe
                                         0, # 30, grita 
                                         0, # 31, atributo de reserva 
                                         0, # 32, atributo de reserva 
                                         0, # 33, atributo de reserva 
                                         0, # 34, atributo de reserva 
                                         0, 
                                         0,
                                         0,
                                         0,
                                         tiemposP,
                                         None
                                         ]
                            
                            # MUTACION
                            atributosP_hijo2, numprocesadores_hijo2 = mutacion (presa_hijo2[13],presa_hijo2[17])
                            
                            # ACTUALIZACION DE LA presa PARA ENTRAR EN LA SIGUIENTE GENERACION
                            presa_hijo2[13]=atributosP_hijo2
                            presa_hijo2[17]=numprocesadores_hijo2
                            #print('el gemelo ha nacido con', len(presa_hijo2[13]) ,'atributos y', presa_hijo2[17] ,'procesadores')
                            
                            for i in range(23,39):  # Campos entre 23 y 38 (inclusive)
                                if i not in atributosP_hijo2:
                                    presa_hijo2[i] = 0 
                                else:
                                    presa_hijo2[i] = np.random.uniform(0,0.5)
                                atributos_hijo2.append(presa_hijo2[i])    
                                                                                   
                            presashijos.append(presa_hijo2)

                        # el padre se va a por tabaco
                        presas[padreP][1] = np.random.uniform(x_min,x_max)
                        presas[padreP][2] = np.random.uniform(y_min,y_max)
    
                    else: # no ha habido ligoteo y el padre busca a la madre
                        # PADRE CORTEJA, les va a costar energía 
                        presas[padreP][8] = reglatrofica / pasostiempo
                        presas[PPmin][8] = reglatrofica / pasostiempo
                        
                        dx = presas[PPmin][1] - presas[padreP][1]
                        dy = presas[PPmin][2] - presas[padreP][2]
                        alfa = degrees(atan2(dy,dx)) - (presas[PPmin][3] - presas[padreP][3]) # ambos estan rotando
                        # si se encuentran, ligan
                        alphaP = alfa * np.pi / 180
                        while alphaP > np.pi * 2:
                            alphaP=alphaP - np.pi * 2 # si hay mas de una vuelta de mas 
                        while alphaP < -np.pi * 2:
                            alphaP=alphaP + np.pi * 2 # si es al reves, tambien
                        
                        # enfilada la proxima presa, va a por ella
                        presas[padreP][1]=presas[padreP][1] + presas[padreP][4] * cos(radians(alphaP))
                        presas[padreP][2]=presas[padreP][2] + presas[padreP][4] * sin(radians(alphaP))
                        # ella busca a quien le pretende 
                        presas[PPmin][1]=presas[PPmin][1] + presas[PPmin][4] * cos(radians(alphaP+np.pi*2))
                        presas[PPmin][2]=presas[PPmin][2] + presas[PPmin][4] * sin(radians(alphaP+np.pi*2))    
                    
                    # Aplicar control de frontera
                    controlfronteraP(presas[padreP])
                    controlfronteraP(presas[PPmin])
                    
        # bucle de reproduccion hasta restablecer el numeroD de depredadores
        if numdepredadoreshijos + SVD <= numeroD:# SVD mayor o igual que numpadresD
            # LOS depredadores SALEN DE LIGUE    (solo adultos)               
            for padreD,depredador in enumerate(depredadores):
                distanciaDD = []
                for madreD,depredador in enumerate(depredadores):
                    if padreD == madreD:
                        PMD = 5
                    else:    
                        PMD=sqrt((depredadores[padreD][1]-depredadores[madreD][1])**2 + (depredadores[padreD][2]-depredadores[madreD][2])**2)   
                        distanciaDD.append(PMD)      
               
                DDmin=np.argmin(np.array(distanciaDD)) # novia mas proxima
                
                if distanciaDD[DDmin] <= 0.1: # si se encuentran, se enamoran
                    depredadores[padreD][7] += 1 # contador de eventos reproductivos
                    depredadores[DDmin][7] += 1               
                    numdepredadoreshijos += 1 # primer cachorro  
                    depredadores[padreD][8] -= 1 / pasostiempo # gasta energia en reproducirse
                    depredadores[DDmin][8] -= 1 / pasostiempo  # gasta energia
                    
                    # depredadores que se reproducen  
                    atributosD_hijo1 = []
                    atributosD_hijo1 = hijo(depredadores[padreD][13],depredadores[DDmin][13])
                    
                    # ahora con el numero de procesadores de la capa oculta
                    Np = depredadores[padreD][17]
                    Mp = depredadores[DDmin][17]
                    # tamaño más próximo por arriba a la media de N y M
                    numprocesadores_hijo1 = int(np.round((Np + Mp) / 2))

                    atributos_hijo1 = []
                    exitostrainD = [] # lista vacia de dos valores con un numero variable de entradas
                    exitostestD = []
                    exitostestRD = []
                    tiemposD = [] # lista vacia de un valor con un numero variable de entradas
                    
                    identificadorD += 1
                    depredador_hijo1 = [identificadorD,
                                        depredadores[padreD][1],# posicion aleatoria de cada depredador
                                        depredadores[DDmin][2],
                                        np.random.uniform(0,drmaxD),# angulo
                                        np.random.uniform(0,velocidadD),# velocidad
                                        [],# 5, juegos
                                        [],# 6, contador de eventos de caza
                                        0,# 7, contador de eventos reproductivos                  
                                        reglatrofica,# 8, fitness al nacer                  
                                        0, # 9, edad
                                        exitostestD,
                                        exitostestRD,
                                        exitostrainD, # 12, tabla de angulos y velocidades con exito (entrenamiento)
                                        atributosD_hijo1, # 13, tabla de angulos y velocidades que no han conseguido atrapar a la presa (entrenamiento)
                                        [], # 14, contador de entrenamientos
                                        0, # 15, contador de conexiones
                                        atributos_hijo1,
                                        numprocesadores_hijo1,
                                        [],
                                        # los pesos no se heredan (de Lamarck a Darwin)
                                        [],
                                        [],
                                        [],
                                        0, # tiempo de decision
                                        0,# 23, agilidad
                                        0,# 24, agresividad                   
                                        0, # 25, agarra
                                        0, # 26, muerde 
                                        0, # 27, tamaño de garras y dientes
                                        0, # 28, salta
                                        0, # 29, al cuello
                                        0, # 30, ruge y acojona 
                                        0, # 31, atributo de reserva 
                                        0, # 32, atributo de reserva 
                                        0, # 33, atributo de reserva 
                                        0, # 34, atributo de reserva 
                                        0, 
                                        0,
                                        0,
                                        0,
                                        tiemposD,
                                        None                                            
                                        ]
                    
                    # MUTACION
                    atributosD_hijo1, numprocesadores_hijo1 = mutacion (depredador_hijo1[13],depredador_hijo1[17])

                    # ACTUALIZACION PARA ENTRAR EN LA SIGUIENTE GENERACION
                    depredador_hijo1[13] = atributosD_hijo1
                    depredador_hijo1[17] = numprocesadores_hijo1
                    #print('el cachorro ha nacido con', len(depredador_hijo1[13]) ,'atributos y', depredador_hijo1[17] ,'procesadores')
                    
                    for i in range(23,39):  # Campos entre 23 y 39 (inclusive)
                        if i not in atributosD_hijo1:
                            depredador_hijo1[i] = 0
                        else:
                            depredador_hijo1[i] = np.random.uniform(0.5,1)
                        atributos_hijo1.append(depredador_hijo1[i])
                    # añadimos el nuevo depredador                                                        
                    depredadoreshijos.append(depredador_hijo1)

                    # si está bien alimentado, puede permitirse un segundo cachorro    
                    if depredadores[DDmin][8] <= depredadores[padreD][8] : 
                        # Eliminar valores repetidos en ambos padres
                        numdepredadoreshijos += 1 # 2 cachorros por generacion
                        depredadores[padreD][8] -= 1 / pasostiempo # gasta energia en reproducirse
                        depredadores[DDmin][8] -= 1 / pasostiempo  # gasta energia
                        
                        atributosD_hijo2 = []
                        atributosD_hijo2 = hijo(depredadores[padreD][13],depredadores[DDmin][13])
                        
                        # ahora con el numero de procesadores de la capa oculta
                        Np = depredadores[DDmin][17]
                        Mp = depredadores[padreD][17]
                        # tamaño más próximo por arriba a la media de N y M
                        numprocesadores_hijo2 = int(np.round((Np + Mp) / 2))  

                        atributos_hijo2 = []
                        exitostrainD = [] # lista vacia de dos valores con un numero variable de entradas
                        exitostestD = []
                        exitostestRD = []
                        tiemposD = [] # lista vacia de un valor con un numero variable de entradas
                        # identificar
                        identificadorD += 1
                        depredador_hijo2 = [identificadorD,
                                            depredadores[DDmin][1],# posicion aleatoria de cada depredador
                                            depredadores[padreD][2],
                                            np.random.uniform(0,drmaxD),# angulo
                                            np.random.uniform(0,velocidadD),# velocidad
                                            [],# juegos
                                            [], # 6, contador de eventos de caza
                                            0, # 7, contador de eventos reproductivos                  
                                            reglatrofica, # 8, fitness al nacer                  
                                            0, # 9, edad 
                                            exitostestD,
                                            exitostestRD,
                                            exitostrainD, # 12, tabla de angulos y velocidades con exito (entrenamiento)
                                            atributosD_hijo2, # 13, tabla de angulos y velocidades que no han conseguido atrapar a la presa (entrenamiento)
                                            [], # 14, contador de entrenamientos
                                            0, # 15, contador de conexiones
                                            atributos_hijo2,
                                            numprocesadores_hijo2,
                                            [],
                                            # los pesos no se heredan (de Lamarck a Darwin)
                                            [],
                                            [],
                                            [],
                                            0, # tiempo de decision
                                            0,# 23, agilidad
                                            0,# 24, agresividad                   
                                            0, # 25, agarra
                                            0, # 26, muerde 
                                            0, # 27, tamaño de garras y dientes
                                            0, # 28, salta
                                            0, # 29, al cuello
                                            0, # 30, ruge y acojona                                            
                                            0, # 31, atributo de reserva 
                                            0, # 32, atributo de reserva 
                                            0, # 33, atributo de reserva 
                                            0, # 34, atributo de reserva 
                                            0, 
                                            0,
                                            0,
                                            0,
                                            tiemposD,
                                            None
                                            ]
                        
                        # MUTACION
                        atributosD_hijo2, numprocesadores_hijo2 = mutacion (depredador_hijo2[13],depredador_hijo2[17])
                        
                        # ACTUALIZACION DE LA presa PARA ENTRAR EN LA SIGUIENTE GENERACION
                        depredador_hijo2[13] = atributosD_hijo2
                        depredador_hijo2[17] = numprocesadores_hijo2
                        #print('el gemelo ha nacido con', len(depredador_hijo2[13]) ,'atributos y', depredador_hijo2[17] ,'procesadores')
                        
                        for i in range(23,39):  # Campos entre 23 y 39 (inclusive)
                            if i not in atributosD_hijo2:
                                depredador_hijo2[i] = 0
                            else:
                                depredador_hijo2[i] = np.random.uniform(0.5,1)
                            atributos_hijo2.append(depredador_hijo2[i])
                        
                        # añadimos el hermano gemelo                                                        
                        depredadoreshijos.append(depredador_hijo2)  

                    # el padre se va a por tabaco
                    depredadores[padreD][1] = np.random.uniform(x_min,x_max)
                    depredadores[padreD][2] = np.random.uniform(y_min,y_max)
                    
                else:
                    # PADRE PERSIGUE
                    depredadores[DDmin][8] = 1 / pasostiempo
                    depredadores[padreD][8] = 1 / pasostiempo
                    
                    dx = depredadores[DDmin][1] - depredadores[padreD][1]
                    dy = depredadores[DDmin][2] - depredadores[padreD][2]
                    alfa = degrees(atan2(dy,dx)) - (depredadores[DDmin][3] - depredadores[padreD][3]) # ambos estan rotando
                    # si se encuentran, ligan
                    alphaD = alfa * np.pi / 180
                    while alphaD > np.pi * 2:
                        alphaD = alphaD - np.pi * 2 
                    while alphaD < -np.pi * 2:
                        alphaD= alphaD + np.pi * 2 # el paso de tiempo es 1, así que la la distancia de la carrera es la velocidadD=depredador[4]
                    
                    # enfilada la proxima novia, va a por ella
                    depredadores[padreD][1] = depredadores[padreD][1] + depredadores[padreD][4]*cos(radians(alphaD))
                    depredadores[padreD][2] = depredadores[padreD][2] + depredadores[padreD][4]*sin(radians(alphaD))

                    depredadores[madreD][1] = depredadores[madreD][1] + depredadores[madreD][4]*cos(radians(alphaD+np.pi*2))
                    depredadores[madreD][2] = depredadores[madreD][2] + depredadores[madreD][4]*sin(radians(alphaD+np.pi*2))
            
                # Aplicar control de frontera
                controlfronteraD(depredadores[padreD])
                controlfronteraD(depredadores[DDmin])
    
        # al final del celo los hijos se independizan de los padres y se esparcen
        for P,presa in enumerate(presashijos):
            presa[1] = np.random.uniform(x_min,x_max)# posicion aleatoria de cada presa           
            presa[2] = np.random.uniform(y_min,y_max) 
        for D,depredador in enumerate(depredadoreshijos):
            depredador[1] = np.random.uniform(x_min,x_max)# posicion aleatoria de cada presa           
            depredador[2] = np.random.uniform(y_min,y_max) 
            
        # Aplicar control de frontera
        for D, depredador in enumerate(depredadores):
            controlfronteraD(depredador)
        for P, presa in enumerate(presas):
            controlfronteraP(presa)
                
        # GRAFICO (llama a la funcion dibujar) PARA CADA PASO  
        #dibujarreproduccion(depredadores,depredadoreshijos,presas,presashijos,generacion,tiempo)
                 # devuelve grafico png
                 
    # de las presas y depredadores que han tomado alguna decision (las habrá que 
    # pasan sin pena ni gloria), nos interesa la media de tiempo en decidir que
    # cada presa y depredador adulto gasta en cada generacion
    for P,presa in enumerate(presas):
        presa[9] += 1 # cumpleaños
        if presas[P][39]: # si no esta vacia (las crias la tienen vacia)
            presas[P][22] = st.mean(presa[39]) # media de millonesimas sg en decidir                          

    for D,depredador in enumerate(depredadores):
        depredador[9] += 1
        if depredadores[D][39]: # si es un hijo, todavia no tiene experiencia
            depredadores[D][22] = st.mean (depredador[39])  # media para decidir en millonesimas  

    for P,presa in enumerate(presas):
        presa[39] = []
        resumengenP.append(lista_a_diccionario([generacion, presa[0], len(presa[6]), presa[8], presa[7], presa[9], len(presa[13]), len(presa[14]), presa[15], presa[17], presa[22], presa[23], presa[24], presa[25], presa[26], presa[27], presa[28], presa[29], presa[30], presa[31], presa[32], presa[33], presa[34], presa[35], presa[36], presa[37], presa[38]]))

    for D, depredador in enumerate(depredadores):
        depredador[39] = [] # no necesitamos conservar cada uno de los tiempos de decision, sino su media              
        resumengenD.append(lista_a_diccionario([generacion, depredador[0], len(depredador[6]), depredador[8], depredador[7], depredador[9], len(depredador[13]), len(depredador[14]), depredador[15], depredador[17], depredador[22], depredador[23], depredador[24], depredador[25], depredador[26], depredador[27], depredador[28], depredador[29], depredador[30], depredador[31], depredador[32], depredador[33], depredador[34], depredador[35], depredador[36], depredador[37], depredador[38]]))
    
    print('generation',generacion)
    print('predators',len(depredadores))
    print('cubs', len(depredadoreshijos))
    print('preys',len(presas))
    print('offspring', len(presashijos))

    #seguimos dentro de una generacion: adultos   
    evolucionpoblaciones.append([generacion,len(presas),criasinviables,numpresashijos,len(devoradasP),presasenfermas,len(depredadores),cachorrosinviables,numdepredadoreshijos,len(dep),depredadoresenfermos,carrona,cazza,nocaza,persecucion])
    
    # al final de la generacion
    
    presas.extend(presashijos) # al acabar el celo y el aprendizaje, todos son iguales
    depredadores.extend(depredadoreshijos)

    #print('comenzaremos la generacion siguiente con',len(depredadores), 'depredadores y',len(presas), 'presas')
    
    guardar_listas_generacion(generacion, depredadores, presas, evolucionpoblaciones, resumengenD, resumengenP, folder_path)
    
# vector de intereses-atributos-argumentos-valores modales de la ultima generacion
print('poderes de los depredadores', contarno0D(depredadores))    
print('intereses de las presas', contarno0P(presas)) 

# hemos ido guardando en una carpeta los 3 ficheros siguientes, que vamos a graficar   
for generacion in range(1,generaciones):
    depredadores, presas, evolucionpoblaciones, resumengenD, resumengenP = cargar_listas_generacion(generacion, folder_path)

generaciones = [item[0] for item in evolucionpoblaciones]
cantidad_presas = [item[1] for item in evolucionpoblaciones]
presas_cazadas = [item[4] for item in evolucionpoblaciones]
presas_enfermas = [item[5] for item in evolucionpoblaciones]
crias_nacidas = [item[3] for item in evolucionpoblaciones]                  
crias_descartadas = [item[2] for item in evolucionpoblaciones]
cantidad_depredadores = [item[6] for item in evolucionpoblaciones]
depredadores_desfallecidos = [item[9] for item in evolucionpoblaciones]
depredadores_enfermos = [item[10] for item in evolucionpoblaciones]
cachorros_nacidos = [item[8] for item in evolucionpoblaciones]
cachorros_descartados = [item[7] for item in evolucionpoblaciones]

carrona = [item[11] for item in evolucionpoblaciones]
cazza = [item[12] for item in evolucionpoblaciones]
nocaza = [item[13] for item in evolucionpoblaciones]
persecucion = [item[14] for item in evolucionpoblaciones]

# evolucion de las poblaciones de depredadores y presas
dibujar_grafica(evolucionpoblaciones) 

# por separado:
graficar_generacion(generaciones, cantidad_presas, 'Generación', 'presas', 'presas adultas')
graficar_generacion(generaciones, cantidad_depredadores, 'Generación', 'depredadores', 'depredadores adultos')
graficar_generacion(generaciones, presas_cazadas, 'Generación', 'presas', 'presas eficientes')
graficar_generacion(generaciones, depredadores_desfallecidos, 'Generación', 'depredadores', 'depredadores fracasados')
graficar_generacion(generaciones, presas_enfermas, 'Generación', 'presas', 'presas ineficientes')
graficar_generacion(generaciones, depredadores_enfermos, 'Generación', 'depredadores', 'malos_depredadores')
graficar_generacion(generaciones, crias_nacidas, 'Generación', 'presas', 'nuevas crias')
graficar_generacion(generaciones, crias_descartadas, 'Generación', 'presas', 'crias desmotivadas')
graficar_generacion(generaciones, cachorros_descartados, 'Generación', 'depredadores', 'depredadores desubicados')
graficar_generacion(generaciones, carrona, 'Generación', 'carroña mano', 'presas heridas')
graficar_generacion(generaciones, cazza, 'Generación', 'encuentros', 'caza')
graficar_generacion(generaciones, nocaza, 'Generación', 'intentos presas', 'presas escapan')
graficar_generacion(generaciones, persecucion, 'Generación', 'intentos depredador', 'intentos_de_caza')

# evolucion de los parametros por cada individuo
graficarevolucion(resumengenD,resumengenP,'eventos_caza') 
graficarevolucion(resumengenD,resumengenP,'alimentacion') 
graficarevolucion(resumengenD,resumengenP,'eventos_reproductivos') 
graficarevolucion(resumengenD,resumengenP,'edad') 
graficarevolucion(resumengenD,resumengenP,'atributos') 
graficarevolucion(resumengenD,resumengenP,'entrenamientos') 
graficarevolucion(resumengenD,resumengenP,'modularidad') 
graficarevolucion(resumengenD,resumengenP,'procesadores') 
graficarevolucion(resumengenD,resumengenP,'tiempo_proceso') 

#    etiquetas = ['generacion', 'identificador', 'eventos_caza', 'alimentacion', 
#'eventos_reproductivos', 'edad', 'atributos', 'entrenamientos', 'modularidad', 
#'procesadores', 'tiempo_proceso', 'mejora_con_el_tiempo','empeora_con_el_tiempo',
#'mejora_con_el_ejercicio','empeora_con_el_ejercicio','mejora_con_el_exito',
#'empeora_con_el_exito','mejora_con_la_alimentacion','empeora_con_la_alimentacion',
#'mejora_con_la_edad_y_la_alimentacion','empeora_con_la edad_y_la_alimentacion',
#'mejora_con_la_edad_y_menos_alimentacion','empeora_con_la_edad_y_menos_alimentacion',
#'mejora_con_la_actividad_reproductora','empeora_con_la_actividad_reproductora',
#'mejora_con_el_numero_de_intereses','empeora_con_el_numero_de_intereses']

# otros por variables:    
#graficarevolucion(resumengenD,resumengenP,'mejora_con_el_tiempo')
#graficarevolucion(resumengenD,resumengenP,'empeora_con_el_tiempo')
#graficarevolucion(resumengenD,resumengenP,'mejora_con_el_ejercicio') 
#graficarevolucion(resumengenD,resumengenP,'empeora_con_el_ejercicio')
#graficarevolucion(resumengenD,resumengenP,'mejora_con_el_exito')    
#graficarevolucion(resumengenD,resumengenP,'empeora_con_el_exito')
#graficarevolucion(resumengenD,resumengenP,'mejora_con_la_alimentacion')
#graficarevolucion(resumengenD,resumengenP,'empeora_con_la_alimentacion')
#graficarevolucion(resumengenD,resumengenP,'mejora_con_la_edad_y_la_alimentacion') 
#graficarevolucion(resumengenD,resumengenP,'empeora_con_la edad_y_la_alimentacion')
#graficarevolucion(resumengenD,resumengenP,'mejora_con_la_edad_y_menos_alimentacion')    
#graficarevolucion(resumengenD,resumengenP,'empeora_con_la_edad_y_menos_alimentacion')
#graficarevolucion(resumengenD,resumengenP,'mejora_con_la_actividad_reproductora')
#graficarevolucion(resumengenD,resumengenP,'empeora_con_la_actividad_reproductora')
#graficarevolucion(resumengenD,resumengenP,'mejora_con_el_numero_de_intereses') 
#graficarevolucion(resumengenD,resumengenP,'empeora_con_el_numero_de_intereses')
 
pass

# medir tiempo
tiempofinal= time.process_time()
tiempototal= (tiempofinal - tiempoinicial)
print('tiempo', tiempototal ,'para', len(generaciones) ,'generaciones y', numeroD ,'depredadores')

# fichero animacion.gif que cuenta la historia
#animacion()
