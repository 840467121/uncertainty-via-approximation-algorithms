# -*- coding: utf-8 -*-

import time
start = time.perf_counter()

import random
import numpy as np 
import pandas as pd
import TDsubfunction as sub
import math
import copy
import scipy.spatial.distance as dist 


global Plist
global Csum



def irred_k_means(Qlist,m,k,C,e,Sum):

# 1. If m = 0
# Assign the points in Q to the
# nearest centers in C.
# Sum = Sum + The clustering cost of Q.
# Return the clustering.
    global Cen
    global Seq
    global CCenter
    global Clu
    if m == 0:
        Qcost = []
        csum = []
        for i in range(0,len(Qlist)):
            dis = []
            for j in range(0,len(C)):
                dis.append(sub.embedding_distance(Qlist[i],C[j]))
            a = dis.index(min(dis))
            Qcost.append(min(dis))
        csum.append(np.array(C))    
        Sum = Sum + sum(Qcost[:len(Qcost)])
        csum.append(Sum)       
        Csum[k-1].append(csum)
        Clu.append(Sum)
        return Sum

# 2. (a) Sample a set S of size O  k α2 from Q.
#    (b) For each set subset S of S of size O  1α do Compute the centroid c of S.  
    b = 1/k
    Ssample = []
    if len(Qlist) > round(4/(b*e)):
        QQlist = random.sample(Qlist,round(4/(b*e)))   
        
        for i in range(round(2/b)):
            Ssample.append(random.sample(QQlist,round(2/e)))
    else:           
        Ssample.append(Qlist) 
           
    for h in range(len(Ssample)):

        Cf = copy.deepcopy(C)
        Cf.append(np.mean(np.array(Ssample[h]), axis=0))

        irred_k_means(Qlist,m-1,k,Cf,e,Sum)

# 3. (a) Consider the points in Q in ascending order of distance from C.
    if m!=k:
        Seq = []       
        Dis = []
        for i in range(0,len(Qlist)):
            dis = []
            for j in range(0,len(C)):
                dis.append(sub.embedding_distance(Qlist[i],C[j]))
            dis = sorted(dis)
            Dis.append(dis[0])
        seq = sorted(Dis) 
        Diss = copy.deepcopy(Dis)
        qlist = copy.deepcopy(Qlist)
        
#    (b) Let U be the first |Q|/2 points in this sequence.
        for i in range(0,len(seq)):
            for j in range(0,len(Diss)):
                if (Diss[j] == seq[i]):
                    Seq.append(qlist[j])           
                    del Diss[j]
                    del qlist[j]
                    break  
        U = Seq[:math.floor(len(Seq)/2)]
        
#    (c) Assign the points in U to the nearest centers in C.
        for i in range(0,len(U)):
            dis = []
            for j in range(0,len(C)):
                dis.append(sub.embedding_distance(U[i],C[j]))
            a = dis.index(min(dis))
            Cen[a].append(U[i])

#    (d) Sum = Sum + The clustering cost of U.
#    (e) Compute the clustering
# 4. Return the clustering which has minimum cost.
        Ucost = sum(seq[:math.floor(len(seq)/2)])
        Sum = Sum + Ucost   
        Qlist = copy.deepcopy(Seq[math.floor(len(Seq)/2):] ) 
        if len(Qlist) < round(4/(b*e)):
            C.append(np.mean(np.array(Qlist), axis=0))
            m = 0
        if Sum < min(Clu):
            irred_k_means(Qlist,m,k,C,e,Sum)


    
def Simplex(P,C):            
    u = C
    U = []
    for i in range(len(u)):
        U.append(u[i])               
    k = len(U)
    n = 3 
    OriginalPoint = U
    Layer_PointNum = np.empty(n)
    Sum_PointNum = np.empty(n)
    sum_pointnum = np.empty(n-1)
    Layer_PointNum[0] = k
    Sum_PointNum[0] = Layer_PointNum[0]
    Layer_PointNum[1] = Layer_PointNum[0] * (Layer_PointNum[0]-1)/2
    Layer_PointNum[2] = Layer_PointNum[0] * Layer_PointNum[1] + Layer_PointNum[1] * (Layer_PointNum[1]-1)/2
    #Layer_PointNum[3] = Layer_PointNum[0] * Layer_PointNum[2] + Layer_PointNum[1] * Layer_PointNum[2] + Layer_PointNum[2] * (Layer_PointNum[2]-1)/2
    #Layer_PointNum[4] = Layer_PointNum[0] * Layer_PointNum[3] + Layer_PointNum[1] * Layer_PointNum[3] + Layer_PointNum[2] * Layer_PointNum[3] + Layer_PointNum[3] * (Layer_PointNum[3]-1)/2
    for i in range (1,n):
        Sum_PointNum[i] = Sum_PointNum[i-1] + Layer_PointNum[i]
        if (i < n): 
            sum_pointnum[i-1] = Sum_PointNum[i] - Layer_PointNum[0]
    PointIndex = []
    pointindex = []
    for k in range (1,n):
        LayerNO = k;
        if (LayerNO == 1):
            for i in range (0,int(Layer_PointNum[0])):      
              for j in range (i+1,int(Layer_PointNum[0])):                          
                      index = [0,i,0,j]
                      PointIndex.append(index)
        else:
            for l in range (0,n-1):  
                if (l != LayerNO-1):
                    if (l < LayerNO-1):
                        for i in range (0,int(Layer_PointNum[l])):      
                          for j in range (0,int(Layer_PointNum[LayerNO-1])):                                  
                                  index = [l,i,LayerNO-1,j]
                                  PointIndex.append(index)
                else:
                    for i in range (0,int(Layer_PointNum[LayerNO-1])):      
                              for j in range (0,int(Layer_PointNum[LayerNO-1])):                              
                                  if (i < j and i != j):    
                                      index = [LayerNO-1,i,LayerNO-1,j]
                                      PointIndex.append(index) 
    for i in range(0,n-1):  
        if(i == 0):
            pointindex.append(PointIndex[0:int(sum_pointnum[i])]) 
        else:
            pointindex.append(PointIndex[int(sum_pointnum[i-1]):int(sum_pointnum[i])])
    Os = []
    pc = []
    pcindex = []
    W = []
    
    for i in range(1,3):
        for j in range(0,len(pointindex[i-1])): 
            bIs = True
            p = 0
            pci = [i,j]
            pcindex.append(pci)
            p = sub.GetSegmentPoint(i,j,pointindex,OriginalPoint,p,bIs) 
            w = sub.weights(p, P, len(P)) 
            W.append(w)   
            os = sub.objectFunction(p, w, P, n=len(P)) 
            Os.append(os)
            s = os             
    
    s = sorted(Os) 
    return pointindex,OriginalPoint,Os,s,pcindex,p,bIs





if __name__ == '__main__':
    P = pd.read_csv("50模型得分1.csv").values
    dataframe = pd.read_csv("50模型得分1.csv",low_memory=False)
    Plist = P.tolist()            
    e = 0.9
    K = 4
    Clu = []
    list1 = []
    Csum = [list(list1) for i in range(K)]
    bestclu = []
   
    for i in range(1,K+1):
        f = [] 
        list2 = [list(list1) for i in range(i)]
        Cen = copy.deepcopy(list2)
        irred_k_means(Plist,i,i,f,e,0)
        minsum = Csum[0][0][-1]
        if len(Csum[i-1])>1:            
            for j in range(len(Csum[i-1])):         
                if minsum > Csum[i-1][j][-1]:
                    minsum = Csum[i-1][j][-1]
                    bestclu = Csum[i-1][j]
    print (bestclu[-2],bestclu[-1])
    C = bestclu[-2]
    list1 = []
    list2 = [list(list1) for i in range(K)]
    Cen = copy.deepcopy(list2)
    for i in range(0,len(Plist)):
                dis = []
                for j in range(0,len(C)):
                    dis.append(sub.embedding_distance(Plist[i],C[j]))
                a = dis.index(min(dis))
                Cen[a].append(P[i])
    Dis = []
    for i in range(0,len(Plist)):
        dis = []
        for j in range(0,len(C)):
            dis.append(sub.embedding_distance(Plist[i],C[j]))
        dis = sorted(dis)
        Dis.append(dis[0])
    Cost = sum(Dis)
    P_pred = []
    for i in range(0,len(P)):
        for j in range(0,len(Cen)):
            for h in range(0,len(Cen[j])):
                if (P[i] == Cen[j][h] ).all():   
                    P_pred.append(j)
    p_pred = np.array(P_pred)
    print(p_pred)
    end1 = time.perf_counter()
    print(end1-start)  

    pointindex,OriginalPoint,Os,s,pcindex,p,bIs = Simplex(P,C)
    for i in range(0, len(Os)):
        if (s[0] == Os[i]):
            pstar = sub.GetSegmentPoint(pcindex[i][0],pcindex[i][1],pointindex,OriginalPoint,p,bIs) 
            print(pstar)
            PSTAR = pd.DataFrame(pstar)
            PSTAR.to_csv("trueValue.csv")
            wresult = sub.weights(pstar, P, n=len(P)) 
            weights = pd.DataFrame(wresult)
            weights.to_csv("weights.csv")    
            break
    end2 = time.perf_counter()
    print(end2-end1)
    print(end2-start)            

