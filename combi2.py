from sys import argv
import queue, threading
import multiprocessing as mp
import concurrent.futures
import csv
import copy
from copy import deepcopy
import sched, time, datetime
import pickle
import gzip
import ubjson
import math
from pathlib import Path
from glob import glob
import numpy as np
import random
import scipy.stats as stats
from random import randint, uniform
from sklearn.metrics import r2_score
#from sklearn.model_selection import cross_val_score
#from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import AdaBoostRegressor,RandomForestRegressor,BaggingRegressor,GradientBoostingRegressor,ExtraTreesRegressor
from sklearn.svm import LinearSVR, SVR, NuSVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
#from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgboost
#import lightgbm as lgb
import indicators as ind
from symbols import *
from math import sqrt
import matplotlib.pyplot as plt
import pydnn

mtt = 3    # months to test
tf = 15
iterations = 10
boundary = 0.38
#starting   = int(60/tf*24*23*39)
#validation = int(60/tf*24*23*mtt)
validation = int(60/tf*24*23*mtt*2)
testing    = int(60/tf*24*23*mtt)
start = 20130101
#test = "_test"
test = ""

#d = "combi_"+str(tf)+"/svr"
d = "combi_"+str(tf)+"/gradient_boosting"
#d = "combi_"+str(tf)+"/neural____network"
#d = "combi_"+str(tf)+"/xg_boosting"

def getSortKey(item):
    global sortKey
    return item[sortKey]

def normalizeX(p,x,f,ts,y,boundaryF=-1):
    global sortKey
    sortKey = 0
    d = []
    factors=[]
    for dd in x: d.append(dd[:])
    try:
        last = len(x[0])
    except:
        print(p+": no record")
    l = len(d)
    if boundaryF<0: ff = boundary
    else          : ff = boundaryF
    for i in  range(last):
        print("normalizing key "+str(i))
        sortKey=i
        d.sort(reverse=True,key=getSortKey)
        if f[i]==1   : fac = abs(d[round(ff/100*l)][i])
        elif f[i]==0 : fac = (abs(d[round(ff/200*l)][i])+abs(d[round((1-ff/200)*l)-1][i]))/2
        elif f[i]==-1: fac = -100
        elif f[i]==24: fac = 24
        #if fac!=0:
        #    for j in range(l): x[j][i] = max(-1,min(1,x[j][i]/fac))
        #elif (fac>450): print("index "+str(i)+" exceeded 450................................")
        fac = 1
        factors.append(fac)
    xx=[]
    tss=[]
    yy=[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    print(p+': sorting done')
    for i in range(len(x)):
        #if (x[i][19]<20):
            xx.append(x[i])
            tss.append(ts[i])
    print(p+': xx tss done')
    for j in range(1):
        for i in range(len(x)):
            #if (x[i][19]<20): 
                yy[j].append(y[j][i])
    return factors,xx,yy,tss


def finalize_training_data(outfile,xf,ts,x,y,bf=-1):
    ff = bf
    if bf==-1: ff = boundary
    #l = len(y)
    #yy = y[:]
    #yy.sort(reverse=True)
    #yf = (abs(yy[round(ff/200*l)])+abs(yy[round((1-ff/200)*l)-1]))/2
    #for i in range(l): y[i] = max(-1,min(1,y[i]/yf))
    w = open(outfile,'wb')
    pickle.dump([xf,1,ts,x,y],w)
    w.close()
    print("written to "+outfile)

def prepare(p,tf,isTrender=False):
    if p=="USATECHIDXUSD": divisor = 1000
    else: divisor = 1
    ema=0
    r = open('data/'+p+str(tf)+'.degap.csv','r')
    r.readline()
    d=0
    while d<start: 
        d,t,o,h,l,c,a,v = ind.parseLine(r,divisor)
    w=[]
    x=[]
    yo=[]
    y=[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    ts=[]
    xfc=[]
    cnt=0
    #if (p[:3]+"USD") in pairs99: p1 = p[:3]+"USD"
    #else: p1 = "USD"+p[:3]
    #if (p[3:]+"USD") in pairs99: p2 = p[3:]+"USD"
    #else: p2 = "USD"+p[3:]    
    for i in range(50):
        d,t,o,h,l,c,a,v = ind.parseLine(r,divisor)
        oo = o
        w.append(o)
    while True:
        xx=[]
        xx.append(c-l)
        if (cnt==0): xfc.append(1)
        xx.append(h-c)
        if (cnt==0): xfc.append(1)
        d,t,o,h,l,c,a,v = ind.parseLine(r,divisor)
        if d==-1: break
        for i in [1,2,3,4,5,6,7,8,10,12,14,16,20,25,30,40,50]:
        #for i in range(1,13):
            xx.append(o-w[50-i])
            if (cnt==0): xfc.append(0)
            #period = i*10
            #ema = ind.indicator(p,"ema",tf,period,d,t,1)
            #if (ema==-999): 
            #    print(p+' skipping '+str(d)+' '+t+"............................................")
            #    break
            #xx.append(ema-o)           
            #if (cnt==0): xfc.append(0)
            #xx.append(ind.indicator(p,"wpr",tf,period,d,t,1))
            #if (cnt==0): xfc.append(-1)

        w = w[1:]
        w.append(o)
        if ema!=-999:
            hr = float(t.split(":")[0])
            if (hr<22 and hr>3): continue
            #print(str(d)+" "+t)
            xx.append(hr)
            if (cnt==0): xfc.append(24)
            x.append(xx)
            for i in range(1): 
                la = ind.indicator(p,"lookahead",tf,0,d,t,i)
                th = commission[p]/unit[p]+spread[p]
                if la > th: la=la
                elif la < -th: la=la
                else: la=la
                y[i].append(la)
            ts.append(str(d)+' '+t)
            cnt+=1
    r.close()
    ind.data={}
    print(p+" closed with "+str(cnt)+ " records...")
    with gzip.open(p+".x", 'wb') as f: pickle.dump([x], f)          
    print(len(x[0]))
    print(len(x))
    xf,x,y,ts  = normalizeX(p,x,xfc,ts,y)
    print(p+": normalization done")
    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)
    print(p+": MinMaxScaler done")
    for i in range(1): 
        finalize_training_data("data/"+p+str(tf)+".s"+str(i),xf,ts,x,y[i])
    print(p+": finalized training data")
    print(len(xf))    
    print(p+" done!")

cache = {}
def cload(currency,file="data/combi15.s",s=0):
    global cache
    print(file in cache)
    if file in cache: 
        d = cache[file]
        print(file+" loaded from cache")
    else:
        r = open(file,'rb')
        d = pickle.load(r)
        r.close()
        #if test=="_test": cache[file] = d
        cache[file] = d
        print(file+" loaded")
    return d[0],d[1],np.array(d[2]),np.array(d[3][s][currency])

def pload(file):
    global cache
    if file in cache: 
        d = cache[file]
        print(file+" loaded from cache")
    else:
        r = open(file,'rb')
        d = pickle.load(r)
        r.close()
        #if test=="_test": cache[file] = d
        #cache[file] = d
        print(file+" loaded")
    return d[0],d[1],d[2],np.array(d[3]),np.array(d[4])

def pload2(file):
    r = open(file,'rb')
    d = pickle.load(r)
    r.close()
    #print(file+" loaded")
    return d[0],d[1],d[2],np.array(d[3]),np.array(d[4])
    
def xg_train(p, trial):
    best_score=-9999999
    best_regressor=None
    
    train=xgboost.DMatrix(xx[p],yy[p])
    ev=xgboost.DMatrix(xt[p],at[p])
    outfile = p+"."+str(trial)+"_"+str(uniform(0.001,0.999))+".regressor"
    print("training "+p+str(trial)+"------------------->"+d+"/"+outfile)
    r = regressor = xgboost.train({'max_depth':1, 'eta':0.1, 'verbosity':0 ,'subsample':0.9,'grow_policy':'lossguide','max_leaves':0}, train,1000, evals=[(ev,'eval')],early_stopping_rounds=10)
    #if not os.path.exists(d): os.makedirs(d)
    #w = open(d+"/"+outfile,'wb')
    #pickle.dump(best_regressor,w)
    #w.close()
    #print(outfile+":  "+str(pf)+":  "+str(m)+":  "+str(bt)+" <<<")
    #w = open(d+"/"+outfile+".accuracy",'w')
    #w.write(str(pf))
    #w.close()    
    #print("written to "+d+"/"+outfile)
    score,m,bt,cnt,pp,rpp,B = simulate(r.predict(xgboost.DMatrix(xtt[p])),att[p],p)
    print("================================================================================================================================"+p+": trial "+str(trial)+": "+str(score*abs(m)))
    return score*abs(m), r

def lgb_train(p, trial, lot=-1):
    best_score=-9999999
    best_regressor=None
    
    train=lgb.Dataset(xx[p],yy[p])
    ev=lgb.Dataset(xtt[p],att[p])
    r =  lgb.train({'num_leaves': 127, 'objective': 'l1', 'boosting':'gbdt', 'num_threads':30, 'verbose':-1,'max_depth':6, 'min_data_in_bin':3000, 'data_seed':randint(0,999999)}, train,1000, valid_sets=[ev], early_stopping_rounds=15,verbose_eval=False)
    best_score,bm,bt,bcnt,pp,rpp,B,bth = simulate(r.predict(xtt[p]),att[p],p,ddd[p])
    if lot>0:print(p+": trial "+str(trial)+": "+str(best_score*lot)+"("+str(round(lot,2))+"): "+str(bcnt)+" / " +str(len(xtt[p])))
    else    :print(p+": trial "+str(trial)+": "+str(best_score*bm)+"("+str(round(bm,2))+"): "+str(bcnt)+" / " +str(len(xtt[p])))
    plt.plot(np.array(range(len(B))), np.array(B), label= '3x')
    plt.savefig(d+"/"+p+"."+str(trial)+"_testing_balance.png")
    plt.close()
    if lot>0: return best_score*lot, best_regressor,lot, bth, bcnt
    else:     return best_score*bm , best_regressor, bm, bth, bcnt


def nn_train(p,trial):
    best_score=-9999999
    best_regressor=None
    l = len(xx[p][0])
    r=regressor = MLPRegressor(warm_start=True,verbose=0,learning_rate_init=0.001,learning_rate='adaptive',hidden_layer_sizes=(l,l),solver='sgd',activation='relu')
    outfile = p+"."+str(trial)+"_"+str(uniform(0.001,0.999))+".regressor"
    print("training "+p+str(trial)+"---------------"+str(l)+" features")
    bj=0
    lr=0.1
    for j in range(1,3000):
        #w = open(d+"/"+outfile+".tmp",'wb')
        #pickle.dump([r,j,bj,best_score,best_regressor],w)
        #w.close()
        r.set_params(max_iter=1,learning_rate_init=lr)
        r.fit(xx[p],yy[p])
        score,m,bt,cnt,pp,rpp,B,th = simulate(r.predict(xt[p]),at[p],p,dd[p])
        if score*abs(m)>best_score:
            lr=lr/2
            bj=j
            bm=m
            best_score = score*abs(m)
            best_regressor = deepcopy(r)
            print(outfile+":"+str(j)+":  "+str(best_score)+":  "+str(bt)+":  "+str(bm)+" <<<")
            plt.plot(np.array(range(len(B))), np.array(B), label= '3x')
            plt.savefig("testing_balance.png")
            plt.close()        
        else:
            print(outfile+":"+str(j)+":  "+str(score*abs(m))+":  "+str(bt)+":  "+str(m)+"......"+str(r.get_params()["learning_rate"]))
        if j-bj>15: break
        #if j>6: 
            #bj=j
            #bm=m
            #best_score = score*abs(m)
            #best_regressor = deepcopy(r)
            #break
    #Path(d+"/"+outfile+".tmp").unlink()
    #if best_score>0:
    if not os.path.exists(d): os.makedirs(d)
    #w = open(d+"/"+outfile,'wb')
    #pickle.dump(best_regressor,w)
    #w.close()
    #w = open(d+"/"+outfile+".accuracy",'w')
    #w.write(str(best_score)+","+str(bm))
    #w.close()    
    score,m,bt,bcnt,pp,rpp,B,bth = simulate(best_regressor.predict(xtt[p]),att[p],p,ddd[p])
    print("================================================================================================================================"+p+": trial "+str(trial)+": "+str(score*abs(m))+": "+str(bcnt)+" / " +str(len(xtt[p])))
    return score*abs(m), best_regressor, m, bth,bcnt


def nn_gpu_train(p,trial):
    best_score=-9999999
    best_regressor=None
    l = len(xx[p][0])
    r=regressor = MLPRegressor(warm_start=True,verbose=0,learning_rate_init=0.1,learning_rate='adaptive',hidden_layer_sizes=(l,l,l,l,l),solver='sgd')
    outfile = p+"."+str(trial)+"_"+str(uniform(0.001,0.999))+".regressor"
    print("training "+p+str(trial)+"---------------"+str(l)+" features")
    bj=0
    for j in range(1,3000):
        #w = open(d+"/"+outfile+".tmp",'wb')
        #pickle.dump([r,j,bj,best_score,best_regressor],w)
        #w.close()
        r.set_params(max_iter=1)
        r.fit(xx[p],yy[p])
        score,m,bt,cnt,pp,rpp,B,th = simulate(r.predict(xt[p]),at[p],p,dd[p])
        if score*abs(m)>best_score:
            bj=j
            bm=m
            best_score = score*abs(m)
            best_regressor = deepcopy(r)
            print(outfile+":"+str(j)+":  "+str(best_score)+":  "+str(bt)+":  "+str(bm)+" <<<")
            #plt.plot(np.array(range(len(B))), np.array(B), label= '3x')
            #plt.savefig(d+"/"+outfile+"_testing_balance.png")
            #plt.close()        
        #else:
            #print(outfile+":"+str(j)+":  "+str(score*abs(m))+":  "+str(bt)+":  "+str(m)+"......"+str(r.get_params()["learning_rate"]))
        if j-bj>10: break

    # build the neural network
    net = pydnn.nn.NN(pre, 'images', 121, 64, rng, pydnn.nn.relu)
    net.add_convolution(72, (7, 7), (2, 2))
    net.add_dropout()
    net.add_convolution(128, (5, 5), (2, 2))
    net.add_dropout()
    net.add_convolution(128, (3, 3), (2, 2))
    net.add_dropout()
    net.add_hidden(3072)
    net.add_dropout()
    net.add_hidden(3072)
    net.add_dropout()
    net.add_logistic()

    # train the network
    lr = pydnn.nn.Adam(learning_rate=pydnn.nn.LearningRateDecay(
                learning_rate=0.006,
                decay=.1))
    net.train(lr)        
    #Path(d+"/"+outfile+".tmp").unlink()
    #if best_score>0:
    if not os.path.exists(d): os.makedirs(d)
    #w = open(d+"/"+outfile,'wb')
    #pickle.dump(best_regressor,w)
    #w.close()
    #w = open(d+"/"+outfile+".accuracy",'w')
    #w.write(str(best_score)+","+str(bm))
    #w.close()    
    score,m,bt,bcnt,pp,rpp,B,bth = simulate(best_regressor.predict(xtt[p]),att[p],p,ddd[p])
    print("================================================================================================================================"+p+": trial "+str(trial)+": "+str(score*abs(m))+": "+str(cnt)+" / " +str(len(xtt[p])))
    return score*abs(m), best_regressor, m, bth,bcnt

def gb_train(p,trial,lot=-1):
    lr = 0.1
    best_score=-9999999
    best_regressor=None
    LV = uniform(0.002,0.01)
    SP = uniform(0.01,0.05)
    SS = 1
    r=GradientBoostingRegressor(verbose=0, warm_start=True, loss='huber', learning_rate=lr, n_estimators=1,max_depth=3,min_samples_leaf=LV,min_samples_split=SP,subsample=SS, alpha=0.9)
    outfile = p+"."+str(trial)+"_"+str(LV)+"_"+str(SP)+"_"+str(SS)+".regressor"
    #print("training "+p+"------------------->"+d+"/"+outfile)
    bj=0
    bm=0
    bcnt=0
    bth=0
    for j in range(1,3000):
        r.set_params(n_estimators=j)
        try:
            r.fit(xx[p],yy[p])
        except:
            print(p+" error")

        score,m,bt,cnt,pp,rpp,B,th = simulate(r.predict(xt[p]),at[p],p,dd[p])
        if score*abs(m)>best_score:
            bj=j
            bm=m
            bth=th
            bcnt=cnt
            best_score = score*abs(m)
            best_regressor = deepcopy(r)
            #print(outfile+":"+str(j)+":  "+str(best_score)+":  "+str(bt)+":  "+str(bm)+" <<<")
            #plt.plot(np.array(range(len(B))), np.array(B), label= '3x')
            #plt.savefig(d+"/"+outfile+"_testing_balance.png")
            #plt.close()        
        #else: print(outfile+":"+str(j)+":  "+str(score*abs(m))+":  "+str(bt)+":  "+str(m)+"......"+str(r.get_params()["learning_rate"]))
        if j-bj>5: break
    #Path(d+"/"+outfile+".tmp").unlink()
    #if best_score>0:
    if not os.path.exists(d): os.makedirs(d)
    #w = open(d+"/"+outfile,'wb')
    #pickle.dump(best_regressor,w)
    #w.close()
    best_score,bm,bt,bcnt,pp,rpp,B,bth = simulate(best_regressor.predict(xtt[p]),att[p],p,ddd[p])
    #w = open(d+"/"+outfile+".accuracy",'w')
    #w.write(str(trial)+","+str(score*abs(m))+","+str(bm))
    #w.close()    
    #print("written to "+d+"/"+outfile)
    if lot>0:print(p+":"+str(len(xx[p]))+":"+str(len(xtt[p]))+": trial "+str(trial)+": "+str(round(best_score*lot))+"("+str(round(lot,2))+"): "+str(bcnt)+" / " +str(len(xtt[p])))
    else    :print(p+":"+str(len(xx[p]))+":"+str(len(xtt[p]))+": trial "+str(trial)+": "+str(round(best_score*bm))+"("+str(round(bm,2))+"): "+str(bcnt)+" / " +str(len(xtt[p])))
    if lot>0: return best_score*lot, best_regressor,lot, bth, bcnt
    else:     return best_score*bm , best_regressor, bm, bth, bcnt

def nearest_train(p,trial,lot=-1):
    lr = 0.1
    best_score=-9999999
    best_regressor=None
    #print("training "+p+"------------------->"+d+"/"+outfile)
    bj=0
    bm=0
    bcnt=0
    bth=0
    for j in range(100,3000,100):
        r=KNeighborsRegressor(n_neighbors=j, weights='uniform', algorithm='auto', leaf_size=30, p=3, metric='minkowski',n_jobs=34)
        try:
            r.fit(xx[p],yy[p])
        except:
            print(p+" error")

        score,m,bt,cnt,pp,rpp,B,th = simulate(r.predict(xt[p]),at[p],p,dd[p])
        if score*abs(m)>best_score:
            bj=j
            bm=m
            bth=th
            bcnt=cnt
            best_score = score*abs(m)
            best_regressor = deepcopy(r)
            print(p+":"+str(j)+":  "+str(best_score)+":  "+str(bt)+":  "+str(bm)+" <<<")
            #plt.plot(np.array(range(len(B))), np.array(B), label= '3x')
            #plt.savefig(d+"/"+outfile+"_testing_balance.png")
            #plt.close()        
        else: print(p+":"+str(j)+":  "+str(score*abs(m))+":  "+str(bt)+":  "+str(m)+"......")
        if j-bj>500: break
    #Path(d+"/"+outfile+".tmp").unlink()
    #if best_score>0:
    if not os.path.exists(d): os.makedirs(d)
    #w = open(d+"/"+outfile,'wb')
    #pickle.dump(best_regressor,w)
    #w.close()
    best_score,bm,bt,bcnt,pp,rpp,B,bth = simulate(best_regressor.predict(xtt[p]),att[p],p,ddd[p])
    #w = open(d+"/"+outfile+".accuracy",'w')
    #w.write(str(trial)+","+str(score*abs(m))+","+str(bm))
    #w.close()    
    #print("written to "+d+"/"+outfile)
    if lot>0:print(p+":"+str(len(xx[p]))+":"+str(len(xtt[p]))+": trial "+str(trial)+": "+str(round(best_score*lot))+"("+str(round(lot,2))+"): "+str(bcnt)+" / " +str(len(xtt[p])))
    else    :print(p+":"+str(len(xx[p]))+":"+str(len(xtt[p]))+": trial "+str(trial)+": "+str(round(best_score*bm))+"("+str(round(bm,2))+"): "+str(bcnt)+" / " +str(len(xtt[p])))
    if lot>0: return best_score*lot, best_regressor,lot, bth, bcnt
    else:     return best_score*bm , best_regressor, bm, bth, bcnt


def svm_train(p,trial,lot=-1):
    best_score=-9999999
    best_regressor=None
    r=LinearSVR(verbose=0,max_iter=3000,tol=0.1)
    #print("training "+p+"------------------->"+d+"/"+outfile)
    bj=0
    bm=0
    bcnt=0
    bth=0
    for j in range(1,2):
        try:
            r.fit(xx[p],yy[p])
        except:
            print(p)

        score,m,bt,cnt,pp,rpp,B,th = simulate(r.predict(xt[p]),at[p],p,dd[p])
        if score*abs(m)>best_score:
            bj=j
            bm=m
            bth=th
            bcnt=cnt
            best_score = score*abs(m)
            best_regressor = deepcopy(r)
            #print(outfile+":"+str(j)+":  "+str(best_score)+":  "+str(bt)+":  "+str(bm)+" <<<")
            #plt.plot(np.array(range(len(B))), np.array(B), label= '3x')
            #plt.savefig(d+"/"+outfile+"_testing_balance.png")
            #plt.close()        
        #else: print(outfile+":"+str(j)+":  "+str(score*abs(m))+":  "+str(bt)+":  "+str(m)+"......"+str(r.get_params()["learning_rate"]))
        if j-bj>10: break
    #Path(d+"/"+outfile+".tmp").unlink()
    #if best_score>0:
    if not os.path.exists(d): os.makedirs(d)
    #w = open(d+"/"+outfile,'wb')
    #pickle.dump(best_regressor,w)
    #w.close()
    best_score,bm,bt,bcnt,pp,rpp,B,bth = simulate(best_regressor.predict(xtt[p]),att[p],p,ddd[p])
    #w = open(d+"/"+outfile+".accuracy",'w')
    #w.write(str(trial)+","+str(score*abs(m))+","+str(bm))
    #w.close()    
    #print("written to "+d+"/"+outfile)
    if lot>0:print(p+":"+str(len(xx[p]))+":"+str(len(xtt[p]))+": trial "+str(trial)+": "+str(best_score*lot)+"("+str(round(lot,2))+"): "+str(bcnt)+" / " +str(len(xtt[p])))
    else    :print(p+":"+str(len(xx[p]))+":"+str(len(xtt[p]))+": trial "+str(trial)+": "+str(best_score*bm)+"("+str(round(bm,2))+"): "+str(bcnt)+" / " +str(len(xtt[p])))
    if lot>0: return best_score*lot, best_regressor,lot, bth, bcnt
    else:     return best_score*bm , best_regressor, bm, bth, bcnt

def rf_train(p,trial):
    lr = 0.03
    best_score=-9999999
    best_regressor=None
    LV = randint(50,500)
    SP = randint(2,13)
    SS = uniform(0.3,0.9)
    r = RandomForestRegressor(warm_start=True,verbose=0,n_jobs=1)
    outfile = p+"."+str(trial)+"_"+str(LV)+"_"+str(SP)+"_"+str(SS)+".regressor"
    print("training "+p+str(trial)+"------------------->"+d+"/"+outfile)
    bj=0
    bm=0
    for j in range(1,322,1):
        r.set_params(n_estimators=j)
        r.fit(xx[p],yy[p])
        score,m,bt,cnt,pp,rpp,B = simulate(r.predict(xt[p]),at[p],p)
        if score*abs(m)>best_score:
            bj=j
            bm=m
            best_score = score*abs(m)
            best_regressor = deepcopy(r)
            print(outfile+":"+str(j)+":  "+str(best_score)+":  "+str(bt)+":  "+str(bm)+" <<<")
            plt.plot(np.array(range(len(B))), np.array(B), label= '3x')
            plt.savefig(d+"/"+outfile+"_testing_balance.png")
            plt.close()        
        else:
            print(outfile+":"+str(j)+":  "+str(score*abs(m))+":  "+str(bt)+":  "+str(m)+"......")
        if j-bj>10: break
    if not os.path.exists(d): os.makedirs(d)
    score,m,bt,cnt,pp,rpp,B = simulate(best_regressor.predict(xtt[p]),att[p],p)
    print("================================================================================================================================"+p+": trial "+str(trial)+": "+str(score*abs(m)))
    return score*abs(m), best_regressor

def simulate(lot,a,pair,ddd):
    sign = 1
    cnt = len(lot)
    balance = 0
    ol = 0
    l = 0
    maxB = -999999
    maxDD = -999999
    tcnt=0
    pcnt=0
    positive=0
    rpositive=0
    B=[]
    tp=0
    tb=0
    #print(lot[:10])
    th = (max(lot)-min(lot))/2*0.3
    #if "IDX" in pair: th = (max(lot)-min(lot))/2*0.00333
    cth = (max(lot)-min(lot))/2*0.03
    switches = 0
    for i in range(4,len(lot)):
        lt = lot[i]
        #print(p+str(lt)+":"+str(a[i]))
        hr = int(ddd[i].split(' ')[1].split(':')[0])
        mn = int(ddd[i].split(' ')[1].split(':')[1])
        if hr == 3: l=0
        elif hr==0 or (hr==1 and mn<30): l=ol
        elif lt>th and hr<=22 : l=1
        elif lt<-th and hr<=22: l=-1
        elif (abs(ol)<0.0000000000000000001 and lt<th and lt>-th) or (ol>0.5 and lt<-cth) or (ol<-0.5 and lt>cth) : l=0
        else: l = ol
        
        if abs(l)>0.00000000001: tcnt+=1
        r=a[i]
        if (r<-990): continue        # rejects -999
        if (r>0 and l>0) or (r<0 and l<0):
            positive+=1
        if abs(l)>0.000000005:
            rpositive+=1
        profit = (l * r - abs(l - ol) * spread[pair] / 2) * unit[pair] / value[pair[-3:]] - abs(l - ol) * commission[pair] / 2
        #profit = (l * r) * unit[pair] / value[pair[-3:]] 
        if profit > 0: pcnt+=1
        if profit > 1600: profit=1600
        if profit <-1600: profit=-1600
        #print(str(ddd[i])+": "+str(l)+"     "+str(profit)+"      "+str(lt)+"     "+str(r)+"      "+str(th))
        #print(xtt[p][i])
        balance += profit
        tp += profit
        if balance > maxB: maxB = balance
        if maxB - balance > maxDD: maxDD = maxB - balance
        B.append(balance)
        if (l<0.5 and ol>0.5) or (l>-0.5 and ol<-0.5): 
            switches = switches +1
            #print(str(ddd[i])+":  "+str(l)+":"+str(ol)+":"+str(tp))
            tp=0
        ol=l
    plt.plot(np.array(range(len(B))), np.array(B), label= '3x')
    plt.savefig(pair+"_testing_balance.png")
    plt.close()        
    #print('threshold:'+str(th))
    #print("switches: "+str(switches))
    #print("balance: "+str(balance))
    if maxDD==0 or cnt<5 or tcnt<5 or switches<int(len(lot)*0.003):
        print(pair+" perfect?" + str(maxDD)+":"+str(cnt)+":"+str(tcnt)+":"+str(switches))
        return balance/1000000000,0.001,0,tcnt,0,0,B,th
    n = len(B)
    return balance, (-100/-maxDD)*sign, float(pcnt)/cnt,switches,fdp(float(positive)/cnt),fdp(float(rpositive)/cnt),B,th

if __name__ == '__main__':
    pairs99 = ["EURUSD","GBPUSD","USDCHF","AUDUSD","NZDUSD"]
    pl=[]
    xf = {}
    yf = {}
    ts = {}
    x = {}
    y = {}
    xx = {}
    xt = {}        
    yy = {}
    yt = {}
    at = {}
    xtt= {}
    att= {}
    dd={}
    ddd={}
    #initialize_last_close(pairs99)
    if argv[1]=="test_i":
        ind.test("CADCHF",60,10,20190903,"07:00:00")
    elif argv[1]=="degap":
        pool = mp.Pool(28)
        for p in pairs99: pl.append(pool.apply_async(degap, (p,tf)))
        #for p in pairs99 +  chosen: pl.append(pool.apply_async(degap, (p,tf)))
        for pl2 in pl: pl2.get()
    elif argv[1]=="prepare":
        pool = mp.Pool(28)
        #pairs99 = ["EURCAD","EURCHF","EURAUD","EURNZD","GBPCAD","GBPCHF","GBPAUD","GBPNZD","CADCHF","AUDCAD","NZDCAD","AUDCHF","NZDCHF","AUDNZD"]
        #for p in pairs99: prepare(p,tf)
        for p in pairs99 :
        #for p in ["EURNZD"]:
        #for p in ["XAUUSD"]:
            if "JPY" in p: continue
            pl.append(pool.apply_async(prepare,(p,tf)))
        for pl2 in pl: pl2.get()
    elif argv[1]=="combi":
        save_combi()
    elif argv[1]=="train":
        tscore = 0

        def evaluate_accuracy(p):
            xf[p],yf[p],ts[p],x[p],y[p] = pload2("data/"+p+str(tf)+".s0")
            #if "JPY" in p: continue
            mm=[]
            ww=[]
            score=0
            for q in range(13):
            #for q in range(12,13):
                mon = int(60/tf*24/4*23)
                last = mon*((14-q)*3)+1
                xx[p] = x[p][q*3*mon:-last]
                xt[p] = x[p][-last:-last+mon*3]
                xtt[p]= x[p][-last+mon*3:-last+mon*6]
                yy[p] = y[p][q*3*mon:-last]
                yt[p] = y[p][-last:-last+mon*3]
                at[p] = []
                att[p] = []
                dd[p] = ts[p][-last:-last+mon*3]
                ddd[p] = ts[p][-last+mon*3:-last+mon*6]
                for i in range(len(ts[p][-last:-last+mon*3])):
                    dt=ts[p][-last:-last+mon*3][i].split(" ")
                    at[p].append(ind.indicator(p,"lookahead",tf,0,dt[0],dt[1],0))
                for i in range(len(ts[p][-last+mon*3:-last+mon*6])):
                    dt=ts[p][-last+mon*3:-last+mon*6][i].split(" ")
                    att[p].append(ind.indicator(p,"lookahead",tf,0,dt[0],dt[1],0))                        
                s, r, m, t, w  = gb_train(p,q)
                #s, r, m, t, w  = lgb_train(p,q)
                #s, r, m, t, w = nn_train(p,q)
                #s, r, m, t, w = nn_gpu_train(p,q)
                #s, r = rf_train(p,q)
                #s, r = xg_train(p,q)
                #s, r, m, t, w  = svm_train(p,q)
                #s, r, m, t, w  = nearest_train(p,q)
                score = score + s
                mm.append(m)
                ww.append(w)
            print("..........................................................................................................."+p+":"+str(score/13)+"...lot determiner")
            lot = (sum(mm)-max(mm)-min(mm))/11
            switches = (sum(ww)-max(ww)-min(ww))/11
            with gzip.open(p+".lot_switches", 'wb') as f:
                pickle.dump([lot,switches], f)
                print("written to "+p+".lot_switches")            
            score=0
            sw=0
            imp = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            iter = 5
            for ii in range(iter):
                for q in range(13):
                #for q in range(12,13):
                    mon = int(60/tf*24/4*23)
                    last = mon*((14-q)*3)+1
                    xx[p] = x[p][q*3*mon:-last]
                    xt[p] = x[p][-last:-last+mon*3]
                    xtt[p]= x[p][-last+mon*3:-last+mon*6]
                    yy[p] = y[p][q*3*mon:-last]
                    yt[p] = y[p][-last:-last+mon*3]
                    at[p] = []
                    att[p] = []
                    dd[p] = ts[p][-last:-last+mon*3]
                    ddd[p] = ts[p][-last+mon*3:-last+mon*6]                    
                    for i in range(len(ts[p][-last:-last+mon*3])):
                        dt=ts[p][-last:-last+mon*3][i].split(" ")
                        at[p].append(ind.indicator(p,"lookahead",tf,0,dt[0],dt[1],0))
                    #for i in range(len(ts[p][-last+mon*3:-last+mon*6])):
                    #    dt=ts[p][-last+mon*3:-last+mon*6][i].split(" ")
                    #    att[p].append(ind.indicator(p,"lookahead",tf,0,dt[0],dt[1],0))
                    for i in range(len(ts[p][-last+mon*3:-last+mon*6])):
                        dt=ts[p][-last+mon*3:-last+mon*6][i].split(" ")
                        att[p].append(ind.indicator(p,"lookahead",tf,0,dt[0],dt[1],0))                        
                    s, r, m, t, w  = gb_train(p,q,lot)
                    #s, r, m, t, w  = lgb_train(p,q,lot)
                    #s, r, m, t, w = nn_train(p,q)
                    #s, r, m, t, w = nn_gpu_train(p,q)
                    #s, r = rf_train(p,q)
                    #s, r = xg_train(p,q)
                    #s, r, m, t, w  = svm_train(p,q,lot)
                    #s, r, m, t, w  = nearest_train(p,q,lot)
                    score = score + s
                    sw = sw + w
                    #for i in range(25): imp[i] = imp[i] + r.feature_importances_[i]
                print(">>>........................................................................................................"+p+":"+str(score/13/(ii+1))+"..."+str(ii)+"...lot:"+str(lot)+"...switches:"+str(switches))
            qq.put([score/13/iter,sw/13/iter,imp])
            #for i in range(25):  imp[i] = imp[i]/12
            #print(imp)


        pr=[]
        qq = mp.Queue()
        sww = 0
        imp2 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        #for p in ["EURCAD","EURNZD","GBPCAD","GBPAUD","GBPNZD","NZDCAD","AUDNZD","XAUUSD", "CHFSGD", "EURNOK", "EURPLN", "EURSEK", "EURSGD", "USDNOK", "USDPLN", "USDSGD", "EURCHF"]:  
        #for p in pairs99 + chosen:  
        #for p in ["EURGBP"]:
        for p in pairs99:
        #for p in ["USDCHF","EURGBP","EURCAD","EURCHF","EURAUD","EURNZD","GBPCAD","GBPCHF","GBPAUD","GBPNZD","CADCHF","NZDCAD","AUDCHF","NZDCHF","AUDNZD","CHFSGD","EURNOK","EURPLN","EURSEK","EURSGD","USDNOK","USDPLN","USDSEK","USDSGD"]:
            if "JPY" in p : continue
            pr.append(mp.Process(target=evaluate_accuracy, args=(p,)))
            pr[-1].start()
            #evaluate_accuracy(p)
            #data = qq.get()
            #score=tscore+data[0]
            #ww = sww+data[1]
            #for i in range(25): imp2[i] = imp2[i]+data[2][i]
            #pr[-1].join()
        for j in range(len(pr)): 
            data = qq.get()
            tscore=tscore+data[0]
            sww = sww+data[1]
            for i in range(25): imp2[i] = imp2[i]+data[2][i]
            print(str(j)+":"+str(tscore)+"------"+str(data[0])+"--------"+str(round(sww)))
        print(str(tscore)+"------"+str(round(sww)))            
        #for i in range(25): imp2[i] = imp2[i]/12/21
        #print(imp2)
        for j in range(len(pr)): pr[j].join()
                
        #pool = mp.Pool(1)
        #for iter in range(iterations):
        #    for p in pairs99:
        #        if "JPY" in p: continue
        #        pl.append(pool.apply_async(nn_train, (shift,p)))
        #for pl2 in pl: pl2.get()
    elif argv[1]=="generate":
        regressors = {}
        lots = {}
        th = {}
        xo = {}
        pr=[]
        qq = mp.Queue()
        sw={}

        def obtain_best(p):
            with gzip.open(p+".lot_switches", 'rb') as f:
                data=pickle.load(f)   

            lot = data[0]
            switches = data[1]
            with gzip.open(p+".x", 'rb') as f: data=pickle.load(f)     

            q = 12
            xo2 = data[0]
            mon = int(60/tf*24/3*23)
            last = mon*((14-q)*3)+1

            xx[p] = x[p][q*3*mon:-last]
            xt[p] = x[p][-last:-last+mon*3]
            xtt[p]= x[p][-last+mon*3:-last+mon*6]
            yy[p] = y[p][q*3*mon:-last]
            yt[p] = y[p][-last:-last+mon*3]
            at[p] = []
            att[p] = []
            dd[p] = ts[p][-last:-last+mon*3]
            ddd[p] = ts[p][-last+mon*3:-last+mon*6]    
            
            #xx[p] = x[p][mon:-mon*3]
            #xt[p] = x[p][-mon*3:]
            #xtt[p] = x[p][-mon*3:]
            #yy[p] = y[p][mon:-mon*3]
            #at[p] = []
            #att[p] = []
            #dd[p] = ts[p][-mon*3:]
            #ddd[p] = ts[p][-mon*3:]                  
            for i in range(len(ts[p][-mon*3:])):
                dt=ts[p][-mon*3:][i].split(" ")
                at[p].append(ind.indicator(p,"lookahead",tf,0,dt[0],dt[1],0))
            for i in range(len(ts[p][-mon*3:])):
                dt=ts[p][-mon*3:][i].split(" ")
                att[p].append(ind.indicator(p,"lookahead",tf,0,dt[0],dt[1],0))
            regressors2=[None, None, None]
            lots2=[None, None, None]
            th2=[0,0,0]
            for i in range(1):
                best_score = -99999
                for j in range(1):
                    s, r, m, t, w = gb_train(p,12)
                    if s>best_score: 
                        best_score = s
                        regressors2[i] = r
                        lots2[i] = m
                        th2[i] = t
                        print(p+":"+str(i)+"/"+str(j)+" :"+str(s)+":"+str(m)+":"+str(w))
            qq.put([p,regressors2,[lot,lot,lot],th2,xo2,switches])

        #for p in pairs99+chosen:
        #for p in ["EURCAD","EURNZD","GBPCAD","GBPAUD","GBPNZD","NZDCAD","AUDNZD","XAUUSD", "CHFSGD", "EURNOK", "EURPLN", "EURSEK", "EURSGD", "USDNOK", "USDPLN", "USDSGD", "EURCHF"]:  
        #for p in ["USDCHF","EURGBP","EURCAD","EURCHF","EURAUD","EURNZD","GBPCAD","GBPCHF","GBPAUD","GBPNZD","CADCHF","NZDCAD","AUDCHF","NZDCHF","AUDNZD","CHFSGD","EURNOK","EURPLN","EURSEK","EURSGD","USDNOK","USDPLN","USDSEK","USDSGD"]:
        #for p in ["USATECHIDXUSD"]:
        for p in pairs99:
            if "JPY" in p: continue
            xf[p],yf[p],ts[p],x[p],y[p] = pload2("data/"+p+str(tf)+".s0")
            pr.append(mp.Process(target=obtain_best, args=(p,)))
            pr[-1].start()

        for i in range(len(pr)): 
            d=qq.get()
            p=d[0]
            regressors[p] = d[1]
            lots[p] = d[2]
            th[p] = d[3]
            xo[p] = d[4]
            sw[p] = d[5]
        for i in range(len(pr)): pr[i].join()

        mx = {}
        mn = {}
        #for p in pairs99+chosen:             
        #for p in ["EURCAD","EURNZD","GBPCAD","GBPAUD","GBPNZD","NZDCAD","AUDNZD","XAUUSD", "CHFSGD", "EURNOK", "EURPLN", "EURSEK", "EURSGD", "USDNOK", "USDPLN", "USDSGD", "EURCHF"]: 
        #for p in ["USDCHF","EURGBP","EURCAD","EURCHF","EURAUD","EURNZD","GBPCAD","GBPCHF","GBPAUD","GBPNZD","CADCHF","NZDCAD","AUDCHF","NZDCHF","AUDNZD","CHFSGD","EURNOK","EURPLN","EURSEK","EURSGD","USDNOK","USDPLN","USDSEK","USDSGD"]:
        #for p in ["USATECHIDXUSD"]:
        for p in pairs99:
            if "JPY" in p: continue
            mx[p] = [0] * 207
            mn[p] = [0] * 207
            for j in range(len(xo[p][0])):
                mxx = -999999
                mnn = 999999
                for i in range(len(xo[p])):
                    if xo[p][i][j]>mxx: mxx=xo[p][i][j]
                    if xo[p][i][j]<mnn: mnn=xo[p][i][j]
                mx[p][j] = mxx
                mn[p][j] = mnn
            print(p+ " fitted")

        with gzip.open("regressors.all", 'wb') as f:
            pickle.dump([regressors,lots,th,sw,mx,mn], f)


    elif argv[1]=="realtime":
        def try_predictor(model, x):
            trees = model.estimators_
            y_pred = model.init_.predict([x])[0]
            #print("*** "+str(y_pred) +  " *** "+ str(model.init_.predict([10]*25)[0]))
            for tree in trees:
                pred = tree[0].predict([x])[0]
                y_pred = y_pred + model.learning_rate*pred  # Summing with LR
            return y_pred


        with gzip.open("regressors.all", 'rb') as f:
            data=pickle.load(f)   
        print("loaded")
        regressors = data[0]
        lots       = data[1]
        th         = data[2]
        sw         = data[3]
        mx         = data[4]
        mn         = data[5]

        def realize(p):
            #print(regressors[p][0].feature_importances_)
            #print(regressors[p][1].feature_importances_)
            #print(regressors[p][2].feature_importances_)
            #continue
            try:
                csvfile = open('porting/'+p+'.input', 'r',newline='')
                lt = list(csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC))
                csvfile.close()
                x = lt[-1]
                if p == "USA30IDXUSD"   : 
                    for ii in range(len(x)-1): x[ii] = ((x[ii]*1000) - mn[p][ii]) / (mx[p][ii] - mn[p][ii])
                    ii=len(x)-1
                    x[ii] = (x[ii] - mn[p][ii]) / (mx[p][ii] - mn[p][ii])
                    c = "US30: "
                elif p == "USA500IDXUSD": 
                    for ii in range(len(x)-1): x[ii] = ((x[ii]*1000) - mn[p][ii]) / (mx[p][ii] - mn[p][ii])
                    ii=len(x)-1
                    x[ii] = (x[ii] - mn[p][ii]) / (mx[p][ii] - mn[p][ii])
                    c = "US500: "
                elif p == "XAUUSD": 
                    for ii in range(len(x)-1): x[ii] = ((x[ii]*100) - mn[p][ii]) / (mx[p][ii] - mn[p][ii])
                    ii=len(x)-1
                    x[ii] = (x[ii] - mn[p][ii]) / (mx[p][ii] - mn[p][ii])
                    c = "XAUUSD: "
                else                    : 
                    for ii in range(len(x)): x[ii] = (x[ii] - mn[p][ii]) / (mx[p][ii] - mn[p][ii])
                    c = p+": "
                for i in range(3):
                    y = regressors[p][i].predict([x])[0] 
                    print(str(y)+" "+str(th[p][i]))
                    #print(try_predictor(regressors[p][i],x))
                    if y> th[p][i]: yy = lots[p][i]
                    if y<-th[p][i]: yy =-lots[p][i]
                    if y< th[p][i] and y>0: yy = 999
                    if y>-th[p][i] and y<0: yy =-999
                    c=c+str(round(yy,3))+"\t"
                    if p == "USA30IDXUSD"   : w = open('/mnt/c/Users/Intel/AppData/Roaming/MetaQuotes/Terminal/Common/Files/US30.'+str(i)+'.output', 'w',newline='')
                    elif p == "USA500IDXUSD": w = open('/mnt/c/Users/Intel/AppData/Roaming/MetaQuotes/Terminal/Common/Files/US500.'+str(i)+'.output', 'w',newline='')
                    else                    : w = open('/mnt/c/Users/Intel/AppData/Roaming/MetaQuotes/Terminal/Common/Files/'+p+'.'+str(i)+'.output', 'w',newline='')
                    w.write(str(yy))
                    w.close()
                c=c+"........................"+str(round(lots[p][0],3))+"\t"+str(round(sw[p]))
                print(c)
            except:
                print(p+": error")

        firstTime = True
        while True:
            time.sleep(3)
            #if (not firstTime) and ((time.localtime().tm_min % 30>0 and time.localtime().tm_min % 30<16) or time.localtime().tm_sec>2): continue
            #if (not firstTime) and time.localtime().tm_sec>2: continue
            firstTime = False
            print("------------------------------------------------------------------------------------------------------------------------------------ "+str(time.localtime().tm_hour)+":"+str(time.localtime().tm_min))
            pr=[]            
            #for p in pairs99+chosen:
            for p in ["GBPAUD","GBPNZD", "CHFSGD", "EURNOK", "EURPLN","EURSGD", "USDNOK","USDSGD"]:
                pr.append(mp.Process(target=realize, args=(p,)))
                pr[-1].start()
                pr[-1].join()
            for i in range(len(pr)): pr[i].join()

    elif argv[1]=="export":
        def export_regressor(p, i, model, l, t, mxx, mnn):
            def wr(w,d): w.write(str(d)+"\r\n")
            def wrtree(w,t):
                wr(w,len(t.feature))
                for i in range(len(t.feature)):
                    wr(w,t.children_left[i])
                    wr(w,t.children_right[i])
                    wr(w,t.feature[i])
                    wr(w,t.threshold[i])
                    wr(w,t.value[i][0][0])
            #w = open('/mnt/c/Users/Intel/AppData/Roaming/MetaQuotes/Terminal/Common/Files/'+p+'.'+str(i)+'.regressor', 'w',newline='')
            w = open('/mnt/c/Users/Lip Phang/AppData/Roaming/MetaQuotes/Terminal/Common/Files/'+p+'.regressor', 'w',newline='')
            wr(w,l)
            wr(w,t)
            wr(w,model.learning_rate)
            wr(w,model.init_.predict([0])[0])
            wr(w,len(model.estimators_))
            for tree in model.estimators_: wrtree(w,tree[0].tree_)
            for j in range(20): wr(w,mnn[j])
            for j in range(20): wr(w,mxx[j])
            print(p+": mn-"+str(len(mnn))+"      : mx-"+str(len(mxx)))
            w.close()

        with gzip.open("regressors.all", 'rb') as f:
            data=pickle.load(f)   
        print("loaded")
        regressors = data[0]
        lots       = data[1]
        th         = data[2]
        sw         = data[3]
        mx         = data[4]
        mn         = data[5]

        #for p in ["EURCAD","EURNZD","GBPCAD","GBPAUD","GBPNZD","NZDCAD","AUDNZD","XAUUSD", "CHFSGD", "EURNOK", "EURPLN", "EURSEK", "EURSGD", "USDNOK", "USDPLN", "USDSGD", "EURCHF"]:
        #for p in ["USDCHF","EURGBP","EURCAD","EURCHF","EURAUD","EURNZD","GBPCAD","GBPCHF","GBPAUD","GBPNZD","CADCHF","NZDCAD","AUDCHF","NZDCHF","AUDNZD","CHFSGD","EURNOK","EURPLN","EURSEK","EURSGD","USDNOK","USDPLN","USDSEK","USDSGD"]:
        #for p in ["USATECHIDXUSD"]:
        for p in pairs99:
            if "JPY" in p: continue
            for i in range(1):  export_regressor(p, i, regressors[p][i], lots[p][i], th[p][i], mx[p], mn[p])
    
    elif argv[1]=="simulate":
        p="EURNZD"
        xf[p],yf[p],ts[p],x[p],y[p] = pload2("data/"+p+str(tf)+".s0")
        with gzip.open(p+".lot_switches", 'rb') as f: data=pickle.load(f)   
        lot = data[0]
        switches = data[1]
        with gzip.open(p+".x", 'rb') as f: data=pickle.load(f)       
        xo2 = data[0]

        
        q = 12
        mon = int(60/tf*24/3*23)
        last = mon*((14-q)*3)+1

        xtt[p]= x[p][-last+mon*3:-last+mon*6]
        at[p] = []
        att[p] = []
        ddd = ts[p][-last+mon*3:-last+mon*6]    
        for i in range(len(ts[p][-mon*3:])):
            dt=ts[p][-mon*3:][i].split(" ")
            at[p].append(ind.indicator(p,"lookahead",tf,0,dt[0],dt[1],0))
        with gzip.open("regressors.all", 'rb') as f: data=pickle.load(f)   
        print("loaded")
        regressors = data[0]
        lots       = data[1]
        th         = data[2]
        sw         = data[3]
        mx         = data[4]
        mn         = data[5]
        score,m,bt,cnt,pp,rpp,B,th  = simulate(regressors[p][0].predict(xtt[p]),at[p],p,ddd)
        print(score*m)
        print(m)
        print(th)
    elif argv[1]=="case":
        xxx = [0.08883432, 0.06857535, 0.40874373, 0.50185874, 0.59737257, 0.59875544, 0.60375306, 0.65758832, 0.62324354, 0.65185156, 0.63796667, 0.70158737, 0.71151698, 0.68060392, 0.69354464, 0.68843949, 0.72235937, 0.75229007, 0.73974192, 0.73175124, 0.69402185, 0.67835282, 0.72020087, 0.59492555, 0.95652174]
        p="GBPAUD"
        xf[p],yf[p],ts[p],x[p],y[p] = pload2("data/"+p+str(tf)+".s0")
        with gzip.open(p+".lot_switches", 'rb') as f: data=pickle.load(f)   
        lot = data[0]
        switches = data[1]
        with gzip.open(p+".x", 'rb') as f: data=pickle.load(f)       
        xo2 = data[0]
        mon = int(60/tf*24*23)
        xtt[p] = x[p][-mon*3:]
        at[p] = []
        ddd = ts[p][-mon*3:]
        for i in range(len(ts[p][-mon*3:])):
            dt=ts[p][-mon*3:][i].split(" ")
            at[p].append(ind.indicator(p,"lookahead",tf,0,dt[0],dt[1],0))
        with gzip.open("regressors.all", 'rb') as f: data=pickle.load(f)   
        print("loaded")
        regressors = data[0]
        lots       = data[1]
        th         = data[2]
        sw         = data[3]
        mx         = data[4]
        mn         = data[5]        
        print(regressors[p][0].predict([xxx])[0])
        print(mx[p][2])
        print(mn[p][2])
