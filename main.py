import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import math
df=pd.read_csv('activos.csv')

def rendimiento(): #Rendimiento del portafolio de inversión
    activos=df.to_numpy(copy=True)
    a, r, rp, ri = [], 0, [], []
    for i in range(11):
        b=[]
        for j in range(1, 11): b.append((activos[i][j]-activos[i+1][j])/activos[i+1][j])
        a.append(b)
    return a #Tabla de rendimientos

rendimientos= rendimiento()
covarianza = np.cov(rendimientos, rowvar=False)

def rendimientoPActivo(): #Rendimientos por activos
    rp = []
    global rendimientos
    for j in range(10):
        r=0
        for i in range(11): r+=rendimientos[i][j]
        rp.append(r/11)
    return rp #Rendimiento promedio 

def rendP(w):
    rn, rp =rendimientoPActivo(), 0
    for n in range(10): rp += (rn[n]*w[n])
    return rp
    
def riesgoP(w):
    global covarianza
    re = 0
    for n in range(10):
        for m in range(10): re += (w[n]*w[m]*covarianza[n][m])
    return re

def fobj(w):
    rendimiento, riesgo = rendP(w), riesgoP(w)
    return rendimiento/riesgo

def restriccion(w):
    suma = sum(w)
    for i in range(10): w[i]=(w[i]/suma)
    return w

def cromosoma():
    w=[random.random() for i in range(10)]
    return restriccion(w)

def crearPoblacion():
    return [cromosoma() for i in range(100)]

poblacion = crearPoblacion()
fitness = []

def calcularFitness():
    global poblacion, fitness
    fitness = []
    for cromosoma in poblacion:
        fitness.append(fobj(cromosoma))

def ordenar():
    global poblacion, fitness
    ordenados = sorted(zip(fitness, poblacion), key = lambda x: x[0], reverse = True)
    poblacion = [cromosoma[1] for cromosoma in ordenados]
    fitness = [j[0] for j in ordenados]

def torneo():
    global poblacion
    h1, h2, h3 = random.randint(0, (len(poblacion)-1)), random.randint(0, (len(poblacion)-1)), random.randint(0, (len(poblacion)-1))
    t = sorted([h1, h2, h3])
    selected = [poblacion[i] for i in t] #ordena los mejores 3
    return selected[0], selected[1] #devuelve los mejores 2

def seleccionCruza():
    hijos = []
    global poblacion, fitness
    calcularFitness()
    ordenar()
    for i in range(40): #80% de la poblacion se cruza
        c1, c2 = torneo()
        alpha = random.random()
        h1 = [(c1[j]*(1 - alpha)+(c2[j]*alpha)) for j in range(10)]
        h2 = [(c2[j] * (1 - alpha) + (c1[j] * alpha)) for j in range(10)]
        hijos.append(restriccion(h1))
        hijos.append(restriccion(h2))
    return hijos

def mutacion(hijos):
    porcentaje = random.randint(1, 4)
    for i in range(porcentaje):
        cMutar, gMutar = random.randint(0, (len(hijos)-1)), random.randint(0, 9)
        genNuevo = random.random()
        while (genNuevo == hijos[cMutar][gMutar]):
            genNuevo = random.random()
        hijos[cMutar][gMutar] = genNuevo
        hijos[cMutar]=restriccion(hijos[cMutar])
    return hijos

def reemplazo(hijos):
    global poblacion
    j = 0
    for i in range(40, 100):
        poblacion[i] = hijos[j]
        j += 1

historialEvolucion = []
for i in range(1000):
    hijos = seleccionCruza()
    mutacion(hijos)
    reemplazo(hijos)
    calcularFitness()
    ordenar()
    historialEvolucion.append(fitness[0])

plt.plot([x for x in range(len(historialEvolucion))], historialEvolucion)
plt.show()
empresas = ['HYUNDAI', 'HONDA', 'GM', 'VOLSKWAGEN', 'FORD', 'TESLA', 'MAZDA', 'BMW', 'TOYOTA', 'VOLVO']
plt.pie(poblacion[0], labels=empresas)
plt.title('PORTAFOLIO DE INVERSIÓN')
plt.show()
rend=rendP(poblacion[0])
ries=riesgoP(poblacion[0])
plt.bar(['RENDIMIENTO', 'RIESGO'], [rend, ries], color=['green','blue'], alpha=0.3)
plt.title('GRÁFICA DE RIESGO Y RENDIMIENTO FINAL')
plt.show()
print(rend)
print(ries)