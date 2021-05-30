import time
import sys
sys.setrecursionlimit(3000)
import random
import copy
import math

compteur = 0
## Q1
# Structure de graphe

def ajoutSommet(graphe, s):
    graphe[s] = {}

def ajoutArete(graphe, s1, s2, poids):
    graphe[s1][s2] = poids
    graphe[s2][s1] = poids

def extraireAretes(graphe):
    res = []
    for i in range(len(graphe)):
        for key in graphe.keys():
            if(key>i):
                res.append([i, key, graphe[i][key]])
    return res

def extraireAretesSommet(graphe, s):
    return graphe[s]

def poidsArete(graphe, a, b):
    return graphe[a][b]

def poidsGraphe(graphe):
    poids = 0
    for i in graphe.keys():
        for j in graphe[i].keys():
            if(i<j):
                poids+=graphe[i][j] 
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

def triAretes(aretes):
    taille_actuelle = 1
    while taille_actuelle < len(aretes)-1:
        gauche = 0
        while(gauche)<len(aretes)-1:
            milieu = min((gauche + taille_actuelle - 1), (len(aretes)-1))
            droite = 2*taille_actuelle + gauche -1 
            if droite >= len(aretes):
                droite = len(aretes)-1
            fusionAretes(aretes, gauche, milieu, droite)
            gauche = gauche + taille_actuelle*2
        taille_actuelle *= 2

def fusionAretes(aretes, gauche, milieu, droite):
    n1 = milieu-gauche +1
    n2 = droite - milieu
    L = [0]*n1
    R = [0]*n2
    for i in range(n1):
        L[i] = aretes[gauche+i]
    for i in range(n2):
        R[i] = aretes[milieu+i+1]
    i, j, k = 0, 0, gauche 
    while i<n1 and j<n2:
        if L[i][2]>R[j][2]:
            aretes[k] = R[j]
            j+=1
        else:
            aretes[k] = L[i]
            i+=1
        k+=1
    while i<n1:
        aretes[k] = L[i]
        i+=1
        k+=1
    while j<n2:
        aretes[k] = R[j]
        j+=1
        k+=1


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
    triAretes(aretes)
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

def grapheFromKruskal(graphe, krusk):
    res = {}
    for key in graphe.keys():
        res[key] = {}
    for arete in krusk:
        s1 = arete[0]
        s2 = arete[1]
        res[s1][s2] = graphe[s1][s2]
        res[s2][s1] = graphe[s1][s2]
    return res
# Kruskal 2

def kruskal2(graphe):
    res = copy.deepcopy(graphe)
    aretes = extraireAretes(res)
    triAretes(aretes)
    aretes.reverse()
    i = 0
    while i<len(aretes):
        arete = aretes.pop(i)
        a = arete[0]
        b = arete[1]
        tmp0 = res[a][b]
        del res[a][b]
        del res[b][a]
        if not connexe(res):
            aretes.insert(i, arete)
            res[a][b] = tmp0
            res[b][a] = tmp0
            i += 1
    return aretes

def parcours(graphe, sommet, visite):
    visite[sommet] = True
    s = 0
    for key in graphe[sommet].keys():
        
        if not visite[key]:
            parcours(graphe, key, visite)

def connexe(graphe):
    visite = [False]*len(graphe)
    parcours(graphe, 0, visite)
    for i in range(1, len(graphe.keys())):
        if not visite[i]:
            return False
    return True

def grapheFromKruskal2(krusk, nbSommets):
    res = {}
    for i in range(nbSommets):
        res[i] = {}
    for arete in krusk:
        s1 = arete[0]
        s2 = arete[1]
        poids = arete[2]
        res[s1][s2] = poids
        res[s2][s1] = poids
    return res
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
        aretes = extraireAretesSommet(graphe, prochain)
        for sommet in aretes.keys():
            suivant = sommet
            poids = aretes[sommet]
            for i in range(len(file)):
                if suivant == file[i]:
                    if prio[suivant] >= poids:
                        pred[suivant] = prochain
                        prio[suivant] = poids
                        majFile(file, prio, i)
                    break
    return pred

def grapheFromPrim(graphe, prim):
    res = {}
    for i in range(len(prim)):
        res[i]={}
    for i in range(len(prim)):
        j = prim[i]
        if(j!=None):
            res[i][j] = graphe[i][j]
            res[j][i] = graphe[i][j]
    return res

def poidsPrim(graphe, prim):
    poids = 0
    for i in range(len(prim)):
        if prim[i] != None:
            poids += poidsArete(graphe, i, prim[i])
    return poids


# Lecture de fichier

def lireFichier(nomFichier):
    graphe = {}
    with open("graphesTests/"+nomFichier, 'r') as f:
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
def compareAlgo(fichiers):
    for fichier in fichiers:
        graphe = lireFichier(fichier)
        print(fichier)
        tempsAlgos(graphe)

def testsPrim(fichiers):
    for fichier in fichiers:
        graphe = lireFichier(fichier)
        print(fichier)
        temps = 0
        for i in range(100):
            temps -= time.time()
            prim(graphe, 0)
            temps+=time.time()
        print(temps/100)

def testsKruskal(fichiers):
     for fichier in fichiers:
        graphe = lireFichier(fichier)
        print(fichier)
        temps = 0
        for i in range(100):
            temps -= time.time()
            kruskal(graphe)
            temps+=time.time()
        print(temps/100)

def testsKruskal2(fichiers):
     for fichier in fichiers:
        graphe = lireFichier(fichier)
        print(fichier)
        temps = 0
        for i in range(100):
            temps -= time.time()
            kruskal2(graphe)
            temps+=time.time()
        print(temps/100)

def tempsAlgos(graphe):
    tkrusk = 0
    tkrusk2 = 0
    tprim = 0
    
    for i in range(10):
        tkrusk-= time.time()
        krusk = kruskal(graphe)
        tkrusk+= time.time()
        tprim-= time.time()
        prim1 = prim(graphe, 0)
        tprim+= time.time()
        tkrusk2 -= time.time()
        krusk2 = kruskal2(graphe)
        tkrusk2 += time.time()

    

    print("Temps de Kruskal(x10): " + str(tkrusk)+" secondes")
    print("Temps de Kruskal2(x10): " + str(tkrusk2)+" secondes")
    print("Temps de Prim(x10) : " + str(tprim)+" secondes")
    
    ratio = "infini"
    if tprim!=0:
        ratio = str(tkrusk/tprim)
    print("Ratio du temps de Kruskal par rapport à Prim : " + ratio)
    ratio = "infini"
    if tkrusk!=0:
        ratio = str(tkrusk2/tkrusk)
    print("Ratio du temps de Kruskal 2 par rapport à Kruskal : " + ratio)

## Q6
def DMST(graphe, degre):
    #tTotalD =  time.time()
    #init de res
    res = []
    cout = float("inf")
    cmpt = 0
    
    #init des pheromones
    pheromones = {}
    aretes = extraireAretes(graphe)
    poidsMax = 0
    poidsMin = float('inf')
    for i in range(len(aretes)):
        if aretes[i][2]>poidsMax:
            poidsMax = aretes[i][2]
        if aretes[i][2]<poidsMin:
            poidsMin = aretes[i][2]
    aretesPheromone = []
    for i in range(len(aretes)):
        phero = (poidsMax-aretes[i][2])+(poidsMax-poidsMin)
        pheromones[(aretes[i][0], aretes[i][1])] = phero
        pheromones[(aretes[i][1], aretes[i][0])] = phero
    """tInit = time.time()-tTotalD
    tDeplacements = 0
    tArbre = 0
    tmajPheromones = 0
    tDeplacementsPurs = 0
    tArbrePur = 0
    tCalculCoutArbre = 0"""
    #boucle principale
    for i in range(100):
        #tInitD = time.time()
        fourmis = []
        for i in range(len(graphe.keys())):
            fourmi = {}
            for j in range(len(graphe.keys())):
                fourmi[j] = False
            fourmi[len(graphe)] = i
            fourmis.append(fourmi)
        #tInit = time.time()-tInitD

        #tDeplacementsD = time.time()
        obj = len(graphe.keys())//2
        for etape in range(obj):
            if etape == obj//3 or etape == 2*obj//3:
                majPheromones(pheromones, aretesPheromone)
                aretesPheromone = []
            
            for f in fourmis:
                deplacement = deplacer(f, graphe, pheromones)
                if deplacement != None:
                    aretesPheromone.append(deplacement)
            
        majPheromones(pheromones, aretesPheromone)
        #tDeplacements+=time.time() - tDeplacementsD
        
        
        tArbreD = time.time()
        T = construireArbre(graphe, aretes, pheromones, degre)
        coutT = poidsKruskal(graphe, T)
        #tCalculCoutArbre += time.time()-tcalculcoutD
        if coutT < cout:
            graphetmp = grapheFromKruskal(graphe, T)
            if(connexe(graphetmp)):
                cout = coutT
                res = T
                cmpt = 0
            
        else:
            cmpt += 1
        #tArbre+=time.time()-tArbreD
        #tmajPheromonesD = time.time()
        majPheromones(pheromones, res)
        
        if cmpt >= 10:
            for arete in res:
                tmp = pheromones[(arete[0], arete[1])]
                if tmp - 1 <= 0:
                    pheromones[(arete[0], arete[1])] -= 1
                    pheromones[(arete[1], arete[0])]
        #tmajPheromones += time.time()-tmajPheromonesD
    """tTotal = time.time()-tTotalD
    print("\n\ntemps total : "+str(tTotal))
    rapport = (tInit/tTotal)*100
    print("\nInit = " +str(tInit))
    print(" /Rapport temps total :"+str(rapport))
    rapport = (tDeplacements/tTotal)*100
    print("\nDéplacements = " +str(tDeplacements))
    print(" /Rapport temps total :"+str(rapport))
    rapport = (tArbre/tTotal)*100
    print("\nArbre = " +str(tArbre))
    print(" /Rapport temps total :"+str(rapport))
    rapport = (tmajPheromones/tTotal)*100
    print("\nMaj des pheromones = " +str(tmajPheromones))
    print(" /Rapport temps total :"+str(rapport))"""
    return res



def majPheromones(pheromones, aretes):
    for a in aretes:
        pheromones[(a[0], a[1])] += 1
        pheromones[(a[1], a[0])] = pheromones[(a[0], a[1])]



def deplacer(fourmi, graphe, pheromones):
    sommet = fourmi[len(fourmi)-1]
    aretes = extraireAretesSommet(graphe, sommet)
    areteChoisie = None
    desirabilite = {}
    desirTot = 0
    aretesPhero = []
    for key in aretes.keys(): 
        if(fourmi[key]==False):
            aretesPhero.append(key)
            denominateur = aretes[key]
            desirabilite[key] = (pheromones[(key, sommet)]/denominateur)
            desirTot += desirabilite[key]

    
    # Trouver une arete aléatoire de façon pondérée
    r = random.random()*desirTot
    for a in aretesPhero:
        r -= desirabilite[a]
        if r < 0:
            areteChoisie = (sommet, a)
            fourmi[a] = True
            fourmi[len(graphe)] = a 
            break
    
    return areteChoisie


def construireArbre(graphe, aretes, pheromones, degre):
    C, T = [], []
    cmpt = 0
    #tri pheromone  
    aretesPheromones = []
    index = 0
    for i in range(len(aretes)):
        a = [index, 0, 0]
        a[2] = pheromones[(aretes[i][0],aretes[i][1])]
        aretesPheromones.append(a)
        index += 1
    triAretes(aretesPheromones)
    aretesPheromones.reverse()
    nbSommets = len(graphe)
    #union-find
    parent = {}
    rang = {}
    for i in range(len(graphe)):
        set(parent, rang, i)

    #tableau degrés
    degres = [0]*nbSommets
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
                indice = aretesPheromones.pop(0)[0]
                C.append(aretes[indice])
            tri(C)
    return T

def testDMST(fichiers):
    
    for fichier in fichiers:
        graphe = lireFichier(fichier)
        with open("testsDMST/"+fichier+"testsDMST.txt", 'w') as f:
            print(fichier)
            f.write(fichier+'\n')
            f.write(str(poidsPrim(graphe, prim(graphe, 0)))+'\n')
            for i in range(2, 10):
                poids = float('inf')
                temps = 0
                for j in range(5):
                    temps-= time.time()
                    res = DMST(graphe, i)
                    temps += time.time()
                    res = grapheFromKruskal(graphe, res)
                    if(poidsGraphe(res)<poids):
                        poids = poidsGraphe(res)       
                temps/=5
                f.write(str(i)+" "+ str(poidsGraphe(res))+" "+ str(temps)+'\n')
                f.flush()

    
## Main
fichiers = ["crd300.gsb", "crd500.gsb", "crd700.gsb", "crd1000.gsb", "shrd150.gsb", "shrd200.gsb", "shrd300.gsb", "str500.gsb", "str700.gsb", "str1000.gsb", "sym300.gsb", "sym500.gsb","sym700.gsb"]
fichiers = ["crd300.gsb", "crd500.gsb", "shrd150.gsb", "shrd200.gsb", "shrd300.gsb", "str500.gsb", "sym300.gsb", "sym500.gsb"]
fichiers=["crd300.gsb"]
testDMST(fichiers)
#tempsTotal = 0- time.time()
#testDMST(fichiers)
#tempsTotal+=time.time()
#print(tempsTotal)
#compareAlgo(fichiers)
fichiers = ["crd300.gsb", "crd500.gsb", "crd700.gsb", "crd1000.gsb"]
#testsPrim(fichiers)
#testsKruskal(fichiers)
#testsKruskal2(fichiers)
graphe = lireFichier("crd300.gsb")
#DMST(graphe, 5)
#print(prim(graphe, 0), poidsPrim(graphe, prim(graphe, 0)))
#print(grapheFromPrim(graphe,prim(graphe, 0)))
#print(kruskal(graphe), poidsKruskal(graphe, kruskal(graphe)))
#print(grapheFromKruskal(graphe, kruskal(graphe)))
#print(kruskal2(graphe), poidsKruskal(graphe, kruskal2(graphe)))
#print(grapheFromKruskal2(kruskal2(graphe), len(graphe)))
#grapheDMST = grapheFromKruskal(graphe, DMST(graphe,5)) 
#print(grapheDMST, poidsGraphe(grapheDMST))
# print(grapheFromKruskal(graphe, kruskal(graphe)))