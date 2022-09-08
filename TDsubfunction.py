# -*- coding: utf-8 -*-

import math
import numpy as np
from numpy import *
import pandas as pd
from sklearn.cluster import KMeans
import scipy.spatial.distance as dist 
import TDsubfunction as sub

def embedding_distance(feature_1, feature_2):
    dist = np.linalg.norm(feature_1 - feature_2)
    return dist


def calcuDistance(data1, data2):

    distance = 0
    for i in range(len(data1)):
        distance += pow((data1[i]-data2[i]), 2)
    return math.sqrt(distance)



def distanceJ(p1, p2):
    res = 0.0
    Pd = []
    Pd.append(p1)
    Pd.append(p2)
    res = dist.pdist(Pd,'jaccard') 
    return res


def distanceO(p1, p2):
    res = 0.0
    Pd = []
    Pd.append(p1)
    Pd.append(p2)
    for h in range(len(p1)): 
        res += np.linalg.norm(p1[h]-p2[h])
    return res

def Delta(P,x): 
   distance = []
   n = len(P)
   if (x == 1):     
       for i in  range(0,n-2) :
            for j in range(i+1,n-1) :          
                     p1=P[i]
                     p2=P[j]                         
                     distance.append(sub.distanceJ(p1,p2))  
   else :
       for i in  range(0,n-2) :
            for j in range(i+1,n-1) :          
                    p1=P[i]
                    p2=P[j]                         
                    distance.append(sub.distanceO(p1,p2)) 
   delta=max(distance)/min(distance)
   print(max(distance),min(distance),max(distance)/min(distance))  
   return delta

def weights(pStar, P, n): 
    sum_result = 0.0
    W = []
    for i in range(0, n):
        sum_result += math.pow(np.linalg.norm(pStar - P[i]), 2)
        #sum_result += math.pow(distance(pStar, P[i]), 2)
        #print (i)
    for l in range(0, n):
        if((pStar==P[l]).all()):
            w = 3
            W.append(w)
        else:
            w = math.log10(sum_result / math.pow(np.linalg.norm(pStar - P[l]), 2))
            #w = math.log10(sum_result / math.pow(distance(pStar, P[i]), 2))
            W.append(w)
    return W

def objectFunction(pStar, W, P, n): 
    object = []
    for i in range(0, n):
        w = W[i]
        p = math.pow(np.linalg.norm(pStar-P[i]),2)
        #p = math.pow(distance(pStar,P[i]),2)
        ##print type(p), type(w)
        s = float(w)*float(p)
        object.append(s)
    os = 0.0
    for i in range(0, n):
        os += object[i]
    return os 
def half(p1, p2):
    return (p1+p2)/2


def maxmin_distance_cluster(data, Theta):

    maxDistance = 0
    start = 0
    index = start
    k = 0 
    dataNum=len(data)
    distance=np.zeros((dataNum,))
    minDistance=np.zeros((dataNum,))
    classes =np.zeros((dataNum,))
    centerIndex=[index]
    ptrCen=data[0]
    for i in range(dataNum):
        ptr1 =data[i]
        d=distanceJ(ptr1,ptrCen)
        distance[i] = d
        classes[i] = k + 1
        if (maxDistance < d):
            maxDistance = d
            index = i 
    minDistance=distance.copy()
    maxVal = maxDistance
    while maxVal > (maxDistance * Theta):
        k = k + 1
        centerIndex+=[index] 
        for i in range(dataNum):
            ptr1 = data[i]
            ptrCen=data[centerIndex[k]]
            d = distanceJ(ptr1, ptrCen)
            distance[i] = d
            if minDistance[i] > distance[i]:
                minDistance[i] = distance[i]
                classes[i] = k + 1       
        index=np.argmax(minDistance)
        maxVal=minDistance[index]
    return classes,centerIndex
 
def  GetSegmentPoint(nLayerNo,nPointIndex,pointindex,OriginalPoint,p,bIs):
    if not bIs:
          return
    n = len(pointindex);
    if (nLayerNo < 0 or nLayerNo > n):
         bIs= False
         return
    if (nPointIndex < 0 or nPointIndex >= len(pointindex[nLayerNo-1])):
         bIs= False
         return
    Layer_PointIndex = pointindex[nLayerNo-1]
    
    if (len(Layer_PointIndex)==0):
         bIs= False
         return    
    if (Layer_PointIndex[nPointIndex][0] == 0):
         p1 = OriginalPoint[Layer_PointIndex[nPointIndex][1]]
         #print (Layer_PointIndex[nPointIndex][0],Layer_PointIndex[nPointIndex][1])
         bIs = True
    else:
        p1 = 0 
        #print (Layer_PointIndex[nPointIndex][0],Layer_PointIndex[nPointIndex][1])
        p1 = GetSegmentPoint(Layer_PointIndex[nPointIndex][0],Layer_PointIndex[nPointIndex][1],pointindex,OriginalPoint,p1,bIs)
    if not bIs:
         return   
    if (Layer_PointIndex[nPointIndex][2] == 0):
         p2 = OriginalPoint[Layer_PointIndex[nPointIndex][3]]
        # print (Layer_PointIndex[nPointIndex][2],Layer_PointIndex[nPointIndex][3])
         bIs = True
    else:
        p2 = 0
       # print (Layer_PointIndex[nPointIndex][2],Layer_PointIndex[nPointIndex][3]) 
        p2 = GetSegmentPoint(Layer_PointIndex[nPointIndex][2],Layer_PointIndex[nPointIndex][3],pointindex,OriginalPoint,p2,bIs)
    if not bIs:
          return	     
    p = (p1 + p2) / 2;    
    return p
