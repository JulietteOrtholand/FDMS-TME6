#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 16:18:52 2018

@author: 3673760
"""

import pandas as pd
import numpy as np  



def random_bandit(data):
    cumul = 0
    list_cumul = [0]
    for i in range(len(data)):
        n = np.random.randint(6,16)
        cumul += data[n].loc[i]
        list_cumul.append(cumul)
    return list_cumul
        

def staticBest(data):
    n = data.sum()[6:].argmax()
    return data[n].cumsum().tolist()

def optimale(data):
    cumul = 0
    list_cumul = [0]
    for i in range(len(data)):
        cumul += data.loc[i][6:].max()
        list_cumul.append(cumul)
    return list_cumul

def ucb(data):
    info = pd.DataFrame(1.,index = range(6,16),columns = ['mu','s'])
    cumul = 0
    list_cumul = [0]
    for i in range(0,10):
        info.loc[i+6]['mu'] = data.loc[i][i+6] 
    for i in range(10,len(data)):
        B = info['mu']+np.sqrt(2*np.log(i)/info['s'])
        n = B.argmax()
        cumul += data.loc[i][n]
        list_cumul.append(cumul)
        info.loc[n]['mu'] = (info.loc[n]['mu']*info.loc[n]['s'] 
                            + data.loc[i][n])/(info.loc[n]['s']+1)
        info.loc[n]['s'] += 1 
    return list_cumul


def linucb(data):
    cumul = 0
    list_cumul = [0]
    A = {}
    b = {}
    for i in range(0,2):
        p = pd.DataFrame(0,index = range(0,5),columns = [0])
        for a in range(0,2):
            if a not in A:
                A[a] = pd.DataFrame(np.identity(5),index = range(1,6),columns = range(1,6))
                b[a] = pd.DataFrame(0, index = range(1,6), columns = [''])
            A_inv =pd.DataFrame(np.linalg.pinv(A[a].values), A[a].columns, A[a].index)
            theta = A_inv.dot(b[a])
            p.loc[a][0] = theta.transpose().dot(data.loc[i][1:6])
            print(p.loc[a][0])
            
            
                
        
        
    
    
        