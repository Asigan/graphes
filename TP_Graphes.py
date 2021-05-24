import time
import sys
sys.setrecursionlimit(1500)
import random
import copy

## Q1
# Structure de graphe

def ajoutSommet(graphe, s):
    graphe.append([])
        


def ajoutArete(graphe, s1, s2, poids):
    graphe[s1].append((s2, poids))
    graphe[s2].append((s1, poids))

def extraireAretes(graphe):
    res = []
    for i in range(len(graphe)):
        for j in range(len(graphe[i])):
            if(graphe[i][j][0]>i):
                res.append([i, graphe[i][j][0], graphe[i][j][1]])
    return res

def extraireAretesSommet(graphe, s):
    return graphe[s]

def poidsArete(graphe, a, b):
    aretes = extraireAretesSommet(graphe, a)
    poids = -1
    for sommet in aretes:
        if sommet[0]==b:
            poids = sommet[1] 
            break
    return poids

def poidsGraphe(graphe):
    poids = 0
    for i in range(len(graphe)):
        for j in range(i,len(graphe)):
            if poids != float("inf"):
                poids += graphe[i][j]
    return poids

## Q2
# Union-Find

def set(parent, rang, a):
    parent[a] = a
    rang[a] = 0

def find(parent, a):
    if parent[a] != a:
        parent[a] = find(parent, parent[a])
    return parent[a]

def union(parent, rang, a, b):
    aRacine = find(parent, a)
    bRacine = find(parent, b)
    if aRacine != bRacine:
        if rang[aRacine] < rang[bRacine]:
            parent[aRacine] = bRacine
        else:
            parent[bRacine] = aRacine
            if rang[aRacine] == rang[bRacine]:
                rang[aRacine] = rang[aRacine] + 1

# Tri Fusion (pour tableau d'aretes)

def tri(tab):
    n = len(tab)
    if n<2:
        return tab
    else:
        m = n//2
    triTab = fusion(tri(tab[:m]),tri(tab[m:]))
    return triTab 

def fusion(tab1,tab2):
    if tab1 == []:
        return tab2
    elif tab2==[]:
        return tab1
    elif tab1[0][2]<tab2[0][2]:
        return [tab1[0]]+fusion(tab1[1:],tab2)
    else:
        return [tab2[0]]+fusion(tab1,tab2[1:])

# Kruskal

def kruskal(graphe):
    res = []
    parent = {}
    rang = {}
    aretes = extraireAretes(graphe)
    for i in range(len(graphe)):
        set(parent, rang, i)
    aretes = tri(aretes)
    for arete in aretes:
        a, b, c = arete
        if find(parent, a) != find(parent, b):
            res.append((a,b))
            union(parent, rang, a, b)
    return res

def poidsKruskal(graphe, krusk):
    poids = 0
    for arete in krusk:
        poids += poidsArete(graphe, arete[0], arete[1])
    return poids

# Kruskal 2

def kruskal2(graphe):
    res = copy.deepcopy(graphe)
    aretes = extraireAretes(res)
    aretes = tri(aretes)
    aretes.reverse()
    i = 0
    while i<len(aretes):
        arete = aretes.pop(i)
        a = arete[0]
        b = arete[1]
        tmp0 = res[a][b]
        tmp1 = res[b][a]
        res[a][b] = float('inf')
        res[b][a] = float('inf')
        if not connexe(res):
            aretes.insert(i, arete)
            res[a][b] = tmp0
            res[b][a] = tmp1
            i += 1
    return aretes

def parcours(graphe, sommet, visite):
    visite[sommet] = True
    s = 0
    for i in range(len(graphe[sommet])):
        if graphe[sommet][i] != float('inf'):
            if not visite[s]:
                parcours(graphe, s, visite)
        s += 1

def connexe(graphe):
    visite = [False]*len(graphe)
    parcours(graphe, 0, visite)
    for i in range(1, len(graphe)):
        if not visite[i]:
            return False
    return True

# Prim

def majFile(file, prio, index):
    if index>0:
        for i in range(index, 0, -1):
            if prio[file[i]] < prio[file[i-1]]:
                file[i], file[i-1] = file[i-1], file[i]
            else:
                break
    return file


def prim(graphe, s):
    file, prio, pred = [s],[],[]
    for i in range(len(graphe)):
        prio.append(float("inf"))
        pred.append(None)
        if i!=s:
            file.append(i)
    prio[s] = 0

    while file != []:
        prochain = file.pop(0)
        aretesProchain = extraireAretesSommet(graphe, prochain)
        for arete in aretesProchain:
            suivant = arete[1]
            poids = arete[2]
            for i in range(len(file)):
                if suivant == file[i]:
                    if prio[suivant] >= poids:
                        pred[suivant] = prochain
                        prio[suivant] = poids
                        majFile(file, prio, i)
                    break
    return pred

def poidsPrim(graphe, prim):
    poids = 0
    for i in range(len(prim)):
        if prim[i] != None:
            poids += poidsArete(graphe, i, prim[i])
    return poids


# Lecture de fichier

def lireFichier(nomFichier):
    graphe = []
    with open(nomFichier, 'r') as f:
        lignes = f.readlines()
    statut = 0
    for ligne in lignes:
        if ligne.startswith("---"):
            statut += 1
        elif statut == 1:
            ajoutSommet(graphe, int(ligne.split()[0]))
        elif statut == 2:
            arete = ligne.split()
            ajoutArete(graphe, int(arete[0]), int(arete[1]), int(arete[2]))
    return graphe

## Q5
# Comparaison des temps de calcul

def tempsAlgos(graphe):
    tkrusk1 = time.time()
    kruskal(graphe)
    tkrusk2 = time.time()
    tprim1 = time.time()
    prim(graphe, 0)
    tprim2 = time.time()
    tkrusk = tkrusk2 - tkrusk1
    tprim = tprim2 - tprim1
    print("Temps de Kruskal : " + str(tkrusk))
    print("Temps de Prim : " + str(tprim))
    print("Ratio du temps de Kruskal par rapport à Prim : " + str(tkrusk/tprim))


## Q6
def DMST(graphe, degre):
    tTotalD =  time.time()
    #init de res
    res = []
    cout = float("inf")
    cmpt = 0

    #init des fourmis
    fourmis = []
    for i in range(len(graphe)):
        fourmis.append([i])

    #init des pheromones
    pheromones = {}
    aretes = extraireAretes(graphe)
    aretesPheromone = []
    for i in range(len(aretes)):
        pheromones[(aretes[i][0], aretes[i][1])] = 1
        pheromones[(aretes[i][1], aretes[i][0])] = 1
    tInit = time.time()-tTotalD
    tDeplacements = 0
    tArbre = 0
    tmajPheromones = 0
    tDeplacementsPurs = 0
    temps = [0]*4
    #boucle principale
    for i in range(100):
        tDeplacementsD = time.time()
        for etape in range(len(graphe)):
            if etape == len(graphe)/3 or etape == 2*len(graphe)/3:
                majPheromones(pheromones, aretesPheromone)
                aretesPheromone = []
            
            for f in fourmis:
                tDeplacementsPursD = time.time()
                deplacement = deplacer(f, graphe, pheromones, temps)
                tDeplacementsPurs += time.time()-tDeplacementsPursD
                if deplacement != None:
                    aretesPheromone.append(deplacement)
            
        tDeplacements+=time.time() - tDeplacementsD
        tArbreD = time.time()
        majPheromones(pheromones, aretesPheromone)
        T = construireArbre(graphe, aretes, pheromones, degre)
        coutT = poidsKruskal(graphe, T)
        if coutT < cout:
            cout = coutT
            res = T
            cmpt = 0
        else:
            cmpt += 1
        tArbre+=time.time()-tArbreD
        tmajPheromonesD = time.time()
        majPheromones(pheromones, res)
        
        if cmpt >= 10:
            for arete in res:
                tmp = pheromones[(arete[0], arete[1])]
                if tmp - 1 <= 0:
                    pheromones[(arete[0], arete[1])] -= 1
        fourmis = []
        for i in range(len(graphe)):
            fourmis.append([i])
        tmajPheromones += time.time()-tmajPheromonesD
    print("temps total Déplacements: "+str(temps[0]))
    rapport = (temps[1]/temps[0])*100
    print("\nInit Déplacements= " +str(temps[1]))
    print(" /Rapport temps total :"+str(rapport))
    rapport = (temps[2]/temps[0])*100
    print("\nAttribution probas aux arêtes = " +str(temps[2]))
    print(" /Rapport temps total :"+str(rapport))
    rapport = (temps[3]/temps[0])*100
    print("\nSelection arêtes = " +str(temps[3]))
    print(" /Rapport temps total :"+str(rapport))
    tTotal = time.time()-tTotalD
    print("temps total : "+str(tTotal))
    rapport = (tInit/tTotal)*100
    print("\nInit = " +str(tInit))
    print(" /Rapport temps total :"+str(rapport))
    rapport = (tDeplacements/tTotal)*100
    print("\nDéplacements = " +str(tDeplacements))
    print(" /Rapport temps total :"+str(rapport))
    rapport = (tDeplacementsPurs/tDeplacements)*100
    print("\nDéplacements Purs= " +str(tDeplacementsPurs))
    print(" /Rapport temps déplacements total :"+str(rapport))
    rapport = (tArbre/tTotal)*100
    print("\nArbre = " +str(tArbre))
    print(" /Rapport temps total :"+str(rapport))
    rapport = (tmajPheromones/tTotal)*100
    print("\nMaj des pheromones = " +str(tmajPheromones))
    print(" /Rapport temps total :"+str(rapport))
    
    
    return res, cout



def majPheromones(pheromones, aretes):
    for a in aretes:
        pheromones[(a[0], a[1])] += 1
        pheromones[(a[1], a[0])] = pheromones[(a[0], a[1])]



def deplacer(fourmi, graphe, pheromones, temps):
    tTotal = time.time()
    sommet = fourmi[len(fourmi)-1]
    aretes = extraireAretesSommet(graphe, sommet)
    areteChoisie = None
    desirabilite = {}
    temps[1]+= time.time() - tTotal
    tattrib = time.time()
    desirTot = 0
    aretesPhero = []
    for a in aretes: 
        if(a[0] not in fourmi):
            aretesPhero.append(a)
            desirabilite[a[0]] = (pheromones[(a[0], sommet)]/(a[1]))
            desirTot += desirabilite[a[0]]
    temps[2]+=time.time()-tattrib
    
    # Trouver une arete aléatoire de façon pondérée
    tSelect = time.time()
    r = random.random()*desirTot
    for a in aretesPhero:
        r -= desirabilite[a[0]]
        if r < 0:
            areteChoisie = (sommet, a[0])
            fourmi.append(a[0])
            break
    
    temps[3]+=time.time()-tSelect
    
    temps[0]+=time.time()-tTotal
    return areteChoisie


def construireArbre(graphe, aretes, pheromones, degre):
    C, T = [], []
    cmpt = 0

    #tri pheromone
    aretesPheromones = copy.deepcopy(aretes)
    index = 0
    for a in aretesPheromones:
        a[2] = pheromones[(a[0],a[1])]
        a.append(index)
        index += 1
    
    aretesPheromones = tri(aretesPheromones)
    aretesPheromones.reverse()
    nbSommets = len(graphe)


    #union-find
    parent = {}
    rang = {}
    for i in range(len(graphe)):
        set(parent, rang, i)

    #tableau degrés
    degres = []
    for i in range(nbSommets):
        degres.append(0)

    while len(T) != nbSommets - 1 and cmpt < 10000:
        cmpt += 1
        if C != []:
            prochaineArete = C.pop(0)
            a, b, c = prochaineArete
            if find(parent, a) != find(parent, b):
                if degres[a] < degre and degres[b] < degre:
                    T.append((a,b))
                    degres[a] += 1
                    degres[b] += 1
                    union(parent, rang, a, b)
        else:
            for i in range(5*nbSommets):
                if aretesPheromones == []:
                    break
                indice = aretesPheromones.pop(0)[3]
                C.append(aretes[indice])
            tri(C)
    return T

def comparePoids(graphe, d):
    pdmst = DMST(graphe,d)[1]
    pmst = poidsKruskal(graphe, kruskal(graphe))
    print("Poids de d-MST : " + str(pdmst))
    print("Poids de MST : " + str(pmst))
    print("Ratio du poids de d-MST par rapport à MST : " + str(pdmst/pmst))


## Main

graphe = lireFichier("crd300.gsb")

print(DMST(graphe,5))