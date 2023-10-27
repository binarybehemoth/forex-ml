from sys import argv
import queue, threading
import multiprocessing as mp
import concurrent.futures
import csv
import copy
from copy import deepcopy
import sched, time, datetime
import pickle
from pathlib import Path
import numpy as np
import random
import scipy.stats as stats
from random import randint, uniform
#from sklearn.metrics import r2_score
#from sklearn.model_selection import cross_val_score
#from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import AdaBoostRegressor,RandomForestRegressor,BaggingRegressor,GradientBoostingRegressor,ExtraTreesRegressor
from sklearn.svm import LinearSVR
from sklearn.neural_network import MLPRegressor
#from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgboost
import indicators as ind
from symbols import *
import matplotlib.pyplot as plt

tf = 15
iterations = 100
boundary = 0.38
starting   = int(60/tf*24*23*37)
validation = int(60/tf*24*23*1)
testing    = int(60/tf*24*23*0.5)
start = 20170101
#test = "_test"
test = ""

#d = "combi_"+str(tf)+"/neural_network"
d = "combi_"+str(tf)+"/svr"
#d = "combi_"+str(tf)+"/gradient_boosting_l1"
#d = "combi_"+str(tf)+"/xg_boosting_l1"

def getSortKey(item):
    global sortKey
    return item[sortKey]

def normalizeX(x,f,boundaryF=-1):
    global sortKey
    sortKey = 0
    d = []
    factors=[]
    for dd in x: d.append(dd[:])
    last = len(x[0])
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
        if fac!=0:
            for j in range(l): x[j][i] = max(-1,min(1,x[j][i]/fac))
        elif (fac>450): print("index "+str(i)+" exceeded 450................................")
        factors.append(fac)
    return factors,x


def finalize_training_data(outfile,xf,ts,x,y,bf=-1):
    ff = bf
    if bf==-1: ff = boundary
    l = len(y)
    yy = y[:]
    yy.sort(reverse=True)
    yf = (abs(yy[round(ff/200*l)])+abs(yy[round((1-ff/200)*l)-1]))/2
    for i in range(l): y[i] = max(-1,min(1,y[i]/yf))
    w = open(outfile,'wb')
    pickle.dump([xf,yf,ts,x,y],w)
    w.close()
    print("written to "+outfile)

def prepare(p,tf,isTrender=False):
    r = open('data/'+p+str(tf)+'.degap.csv','r')
    r.readline()
    d=0
    while d<start: d,t,o,h,l,c,a,v = ind.parseLine(r)
    w=[]
    x=[]
    yo=[]
    y=[[],[],[],[],[],[],[],[],[],[],[],[]]
    ts=[]
    xfc=[]
    cnt=0
    for i in range(12):
        d,t,o,h,l,c,a,v = ind.parseLine(r)
        oo = o
        w.append(a)
    while True:
        xx=[]
        xx.append(c-l)
        if (cnt==0): xfc.append(1)
        xx.append(h-c)
        if (cnt==0): xfc.append(1)
        d,t,o,h,l,c,a,v = ind.parseLine(r)
        if d==-1: break
        xx.append(o-w[11])
        if (cnt==0): xfc.append(0)
        xx.append(o-w[10])
        if (cnt==0): xfc.append(0)
        xx.append(o-w[9])
        if (cnt==0): xfc.append(0)
        xx.append(o-w[8])
        if (cnt==0): xfc.append(0)
        xx.append(o-w[7])
        if (cnt==0): xfc.append(0)
        xx.append(o-w[6])
        if (cnt==0): xfc.append(0)
        xx.append(o-w[5])
        if (cnt==0): xfc.append(0)
        xx.append(o-w[4])
        if (cnt==0): xfc.append(0)
        xx.append(o-w[3])
        if (cnt==0): xfc.append(0)
        xx.append(o-w[2])
        if (cnt==0): xfc.append(0)
        xx.append(o-w[1])
        if (cnt==0): xfc.append(0)
        xx.append(o-w[0])
        if (cnt==0): xfc.append(0)
        w = w[1:]
        w.append(a)
        #for period in [6,12,18,24,30,40,60,80,100,150,200,400,600,800,1000,1500,2000]:
        for period in [12,18,24,30,40,60,80,100,150,200,400,600,800,1000,1500,2000]:
            ema = ind.indicator(p,"ema",tf,period,d,t,1)
            if (ema==-999): 
                print(p+' skipping '+str(d)+' '+t+"............................................")
                break
            xx.append(ema-o)
            ema2 = ind.indicator(p,"ema",tf,period,d,t,2)
            if (ema2==-999): 
                print(p+' skipping '+str(d)+' '+t+"............................................")
                break
            xx.append(ema-ema2)
            if (cnt==0): xfc.append(0)            
            if (cnt==0): xfc.append(0)
            xx.append(ind.indicator(p,"wpr",tf,period,d,t,1))
            if (cnt==0): xfc.append(-1)
            xx.append(ind.indicator(p,"sd",tf,period,d,t,1))
            if (cnt==0): xfc.append(1)
            xx.append(ind.indicator(p,"bb",tf,period,d,t,1))
            if (cnt==0): xfc.append(0)
            xx.append(ind.indicator(p,"breakout",tf,period,d,t,1))
            if (cnt==0): xfc.append(0)
            xx.append(ind.indicator(p,"cci",tf,period,d,t,1))
            if (cnt==0): xfc.append(0)
            xx.append(ind.indicator(p,"rsi",tf,period,d,t,1))
            if (cnt==0): xfc.append(1)
            xx.append(ind.indicator(p,"atr",tf,period,d,t,1))
            if (cnt==0): xfc.append(1)
            #xx.append(ind.indicator(p,"macd",tf,period,d,t,1))
            #if (cnt==0): xfc.append(0)
            #xx.append(ind.indicator(p,"macd_signal",tf,period,d,t,1))
            #if (cnt==0): xfc.append(0)
            xx.append(ind.indicator(p,"demarker",tf,period,d,t,1))
            if (cnt==0): xfc.append(1)
            #xx.append(ind.indicator(p,"trix",tf,period,d,t,1))
            #if (cnt==0): xfc.append(0)
            #xx.append(ind.indicator(p,"adx",tf,period,d,t,1)[0])
            #if (cnt==0): xfc.append(1)
            #xx.append(ind.indicator(p,"adx",tf,period,d,t,1)[1])
            #if (cnt==0): xfc.append(0)
            xx.append(ind.indicator(p,"momentum",tf,period,d,t,1))
            if (cnt==0): xfc.append(0)
            #xx.append(ind.indicator(p,"rvi",tf,period,d,t,1)[0])
            #if (cnt==0): xfc.append(0)
            #xx.append(ind.indicator(p,"rvi",tf,period,d,t,1)[1])
            #if (cnt==0): xfc.append(0)
        if ema!=-999:        
            #xx.append(ind.indicator(p,"psar",tf,-1,d,t,1))
            #if (cnt==0): xfc.append(0)
            xx.append(float(t.split(":")[0]))
            if (cnt==0): xfc.append(24)
            x.append(xx)
            for shift in range(1): 
                la = ind.indicator(p,"ideal_lot",tf,0,d,t,shift)
                if isTrender:
                    if xx[4]*la > 0: y[shift].append(la)
                    else           : y[shift].append(0)
                else        : y[shift].append(la)
            ts.append(str(d)+' '+t)
            cnt+=1
            oo=o
    r.close()
    ind.data={}
    print(p+" closed with "+str(cnt)+ " records...")
    xf,x  = normalizeX(x,xfc)
    for i in range(1): 
        finalize_training_data("data/"+p+str(tf)+".s"+str(i),xf,ts,x,y[i])
    print(len(xf))    
    print(p+" done!")

cache = {}
def cload(currency,file="data/combi15.s"):
    global cache
    #print(file)
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
    return d[0],d[1],np.array(d[2]),np.array(d[3][currency])

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

def combi(tf):
    global cache

    if not empty("data/combi"+str(tf)+".s"):
        r = open("data/combi"+str(tf)+".s","rb")
        cache["data/combi"+str(tf)+".s"] = pickle.load(r)
        r.close()
        return

    xf = {}
    yf = {}
    ts = {}
    x = {}
    y = {}
    fp = {}
    data = {}
    for p in pairs99 + chosen:
        xf[p],yf[p],ts[p],x[p],y[p] = pload("data/"+p+str(tf)+".s0")
        print(str(len(ts[p]))+" records")
        data[p] = {}
        for i in range(len(ts[p])): data[p][ts[p][i]] = x[p][i]
        fp[p] = open('data/'+str(p)+str(tf)+'.degap.csv','r')
    rxf = []
    rts = []
    rx  = []
    ry  = {}
    for p in pairs99 + chosen: 
        rxf = rxf + xf[p]
        ry[p] = []
    for t in ts["EURUSD"]:
        #if not (t in c):continue
        all_present = True
        for p in pairs99 + chosen:
            if not (t in data[p]):
                all_present = False
                print(p+": "+t+" not present")
                break
        if not all_present: continue
        if len(rx)%1000==0:print(t)
        xx  = []
        for p in pairs99 + chosen:xx = list(xx) + list(data[p][t])
        rts.append(t)
        rx.append(xx)
        for p in pairs99 + chosen:
            ry[p].append(ind.indicator(p,"ideal_lot",tf,0,int(t.split(" ")[0]),t.split(" ")[1],0))
    print(str(len(rxf))+" features")
    print(str(len(rx[0]))+" features")
    print(str(len(ry["EURUSD"]))+" records")
    print("ended at..."+rts[-1])
    #print("saving...")
    #w = open("data/combi"+str(tf)+".s","wb")
    #pickle.dump([rxf,rts,rx,ry],w,protocol=pickle.HIGHEST_PROTOCOL)
    #w.close()
    cache["data/combi"+str(tf)+".s"]=[rxf,rts,rx,ry]
    print("Combi finalized")


def accuracy(actual,predictions):
    hits = 0
    zeroes = 0
    sum = 0
    l = 0
    for i in range(len(actual)):
        if np.isnan(predictions[i]): continue
        l=l+1
        if (actual[i]>1 and predictions[i]>1) or (actual[i]<1 and predictions[i]<1): hits+=1
        if actual[i]*predictions[i]==0: zeroes+=1
        #sum = sum + (actual[i]*(predictions[i]/abs(predictions[i])))
    return round(hits/l,4),sum/l,round(zeroes/l,4)

def mlp_train(currency,outdir,outfile):
    xf,ts,x,y = cload(currency)
    regressor=[]
    xtrain=[]
    xtest=[]
    ytrain=[]
    ytest=[]
    folds=5
    print("training "+currency+"------------------->"+outdir+"/"+outfile)
    r = MLPRegressor(warm_start=True,verbose=2,tol=1e-7,learning_rate='adaptive',hidden_layer_sizes=(len(x[0])),solver='sgd')
    for i in range(folds):
        regressor.append(deepcopy(r))
        xx, xt, yy, yt = train_test_split(x[:-validation],y[:-validation],test_size=1/folds)
        xtrain.append(xx)
        xtest.append(xt)
        ytrain.append(yy)
        ytest.append(yt)
    ne = 1
    bne = 0
    best_score=-9999
    while True:
        score=0
        for i in range(folds):
            regressor[i].set_params(max_iter=ne)
            #regressor.set_params(max_iter=1)
            regressor[i].fit(xtrain[i],ytrain[i])
            sc,ave_gain,zhits = accuracy(ytest[i], regressor[i].predict(xtest[i]))
            score+=sc
            print(outfile+"..."+str(i)+"..."+str(round(score/(i+1),4)))
        score=round(score/5,4)
        if score>best_score:
            best_score = score
            best_regressor = deepcopy(regressor[0])
            bne=ne
            print(outfile+":"+str(ne)+":"+str(score)+ ":"+str(zhits)+" <<<")
        else:
            print(outfile+":"+str(ne)+":"+str(score)+ ":"+str(zhits))
        ne+=1
        if ne-bne>7: break
    best_regressor.fit(x[tStart:-validation],y[tStart:-validation])
    if not os.path.exists(outdir): os.makedirs(outdir)
    w = open(outdir+"/"+outfile,'wb')
    pickle.dump(best_regressor,w)
    w.close()
    w = open(outdir+"/"+outfile+".accuracy",'w')
    w.write(str(best_score))
    w.close()    
    print("written to "+outdir+"/"+outfile)
    return best_score
    regressor = MLPRegressor(warm_start=True,verbose=2,tol=1e-7,learning_rate='adaptive',hidden_layer_sizes=(len(x[0]),len(x[0])),solver='sgd')

def svr_train(currency,outdir,outfile,tStart=-starting):
    xf,ts,x,y = cload(currency)
    regressor=[]
    xtrain=[]
    xtest=[]
    ytrain=[]
    ytest=[]
    folds=5
    print("training "+currency+"------------------->"+outdir+"/"+outfile)
    r=LinearSVR(verbose=2)
    for i in range(folds):
        regressor.append(deepcopy(r))
        xx, xt, yy, yt = train_test_split(x[:-validation],y[:-validation],test_size=1/folds)
        xtrain.append(xx)
        xtest.append(xt)
        ytrain.append(yy)
        ytest.append(yt)
    ne = 1
    bne = 0
    best_score=-9999
    while True:
        score=0
        for i in range(folds):
            #regressor.set_params(max_iter=1)
            regressor[i].fit(xtrain[i],ytrain[i])
            sc,ave_gain,zhits = accuracy(ytest[i], regressor[i].predict(xtest[i]))
            score+=sc
            print(outfile+"..."+str(i)+"..."+str(round(score/(i+1),4)))
        score=round(score/5,4)
        if score>best_score:
            best_score = score
            best_regressor = deepcopy(regressor[0])
            bne=ne
            print(outfile+":"+str(ne)+":"+str(score)+ ":"+str(zhits)+" <<<")
        else:
            print(outfile+":"+str(ne)+":"+str(score)+ ":"+str(zhits))
        ne+=1
        if ne-bne>7: break
        break
    best_regressor.fit(x[tStart:-validation],y[tStart:-validation])
    if not os.path.exists(outdir): os.makedirs(outdir)
    w = open(outdir+"/"+outfile,'wb')
    pickle.dump(best_regressor,w)
    w.close()
    w = open(outdir+"/"+outfile+".accuracy",'w')
    w.write(str(best_score))
    w.close()    
    print("written to "+outdir+"/"+outfile)
    return best_score

def gb_train(pair,outdir,outfile,tStart=-starting):
    def train(regressor,ii,x_train,y_train,x_test,y_test):
        regressor.set_params(n_estimators=ii)
        regressor.fit(x_train,y_train)
        sc = regressor.score(x_test,y_test)
        return sc

    xf,ts,x,y = cload(pair)
    regressor=[]
    xtrain=[]
    xtest=[]
    ytrain=[]
    ytest=[]
    sc=[]
    future=[]
    folds=1
    print("training "+pair+"------------------->"+outdir+"/"+outfile)
    r=GradientBoostingRegressor(verbose=0, warm_start=True, loss='huber', learning_rate=0.01, n_estimators=1,max_depth=13,min_samples_leaf=randint(50,500),min_samples_split=randint(2,13),subsample=uniform(0.1,0.9))
    for i in range(folds):
        regressor.append(deepcopy(r))
        xx, xt, yy, yt = train_test_split(x[tStart:-validation],y[tStart:-validation],test_size=0.2)
        xtrain.append(xx)
        xtest.append(xt)
        ytrain.append(yy)
        ytest.append(yt)
        sc.append(0)
        future.append(0)
    ne = 1
    bne = 0
    best_score=-9999
    while True:
        score=0
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for i in range(folds):
                future[i] = executor.submit(train,regressor[i],ne,xtrain[i],ytrain[i],xtest[i],ytest[i])
            for i in range(folds):                
                sc[i]=future[i].result()
                #print(outfile+"..."+str(i)+"..."+str(round(sc[i],4)))
            executor.shutdown(wait=True)
        score=round(sum(sc)/folds,4)
        if score>best_score:
            best_score = score
            best_regressor = deepcopy(regressor[0])
            bne=ne
            print(outfile+":"+str(ne)+":"+str(score)+" <<<")
        else:
            print(outfile+":"+str(ne)+":"+str(score))
        ne+=1
        if ne-bne>7: break
    best_regressor.fit(x[tStart:-validation],y[tStart:-validation])
    if not os.path.exists(outdir): os.makedirs(outdir)
    w = open(outdir+"/"+outfile,'wb')
    pickle.dump(best_regressor,w)
    w.close()
    w = open(outdir+"/"+outfile+".accuracy",'w')
    w.write(str(best_score))
    w.close()    
    print("written to "+outdir+"/"+outfile)
    return best_score

def xb_train(currency,outdir,outfile):
    xf,ts,x,y = cload(currency)
    print("training "+currency+"------------------->"+outdir+"/"+outfile)
    train=xgboost.DMatrix(x[:-validation],y[:-validation])
    ev=xgboost.DMatrix(x[-validation:-testing],y[-validation:-testing])
    regressor = xgboost.train({'max_depth':13, 'eta':0.1, 'verbosity':0 ,'subsample':0.5,'grow_policy':'lossguide','max_leaves':0}, train,1000, evals=[(ev,'eval')],early_stopping_rounds=9)
    score,ave_gain,zhits = accuracy(y[-validation:-testing],regressor.predict(xgboost.DMatrix(x[-validation:-testing])))
    print(outfile+":"+str(score)+":"+str(zhits)+":"+str(ave_gain))
    if not os.path.exists(outdir): os.makedirs(outdir)
    w = open(outdir+"/"+outfile,'wb')
    pickle.dump(regressor,w)
    w.close()
    w = open(outdir+"/"+outfile+".accuracy",'w')
    w.write(str(score))
    w.close()        
    print("written to "+outdir+"/"+outfile)

def simulate(lot,a,pair,bc=0,bcc=0,sc=0,scc=0,bc2=0,bcc2=0,sc2=0,scc2=0,bc3=0,bcc3=0,sc3=0,scc3=0):
    balance = 0
    ol = 0
    l = 0
    maxB = -999999
    maxDD = -999999
    cnt=0
    tcnt=0
    pcnt=0
    positive=0
    rpositive=0
    B=[]
    #print(y[:10])
    for i in range(0,len(lot)):
        lt = lot[i]
        if ol>=0:
            if lt>ol: l=lt
            if lt<0: l=lt
        if ol<=0:
            if lt<ol: l=lt
            if lt>0: l=lt
        if abs(l)>0.0001: tcnt+=1
        r=a[i]
        if (r>0 and l>0) or (r<0 and l<0):
            positive+=1
        if abs(l)>0.000000005:
            rpositive+=1
        profit = (l * r - abs(l - ol) * spread[pair] / 2) * unit[pair] / value[pair[-3:]] - abs(l - ol) * commission[pair] / 2
        if profit > 0: pcnt+=1
        if profit > 1200: profit=1200
        if profit <-1200: profit=-1200
        balance += profit
        if balance > maxB: maxB = balance
        if maxB - balance > maxDD: maxDD = maxB - balance
        B.append(balance)
        ol=l
    cnt = len(lot)
    if maxDD==0 or cnt==0 or tcnt<5:
        return balance/1000,0.001,0,tcnt,0,0,B
    #if uniform(0,1)<0.001: 
    print(pair+":"+str(int(balance))+":"+str(tcnt)+":\t"+str(bc)+","+str(bcc)+","+str(sc)+","+str(scc)+",\t"+str(bc2)+","+str(bcc2)+","+str(sc2)+","+str(scc2)+",\t"+str(bc3)+","+str(bcc3)+","+str(sc3)+","+str(scc3))        
    return balance, min(1,-100/-maxDD), float(pcnt)/cnt,tcnt,fdp(float(positive)/cnt),fdp(float(rpositive)/cnt),B

def reset_sum():
    w = open("combi_"+str(tf)+"/sum.txt",'w')
    w.write("0")
    w.close()

def test(p,m):
    def add_sum(num):
        r = open("combi_"+str(tf)+"/sum.txt",'r')
        d = list(csv.reader(r, quoting=csv.QUOTE_NONNUMERIC))
        r.close()
        w = open("combi_"+str(tf)+"/sum.txt",'w')
        w.write(str(float(d[0][0])+num))
        w.close()
        print("SUM added: "+str(round(d[0][0]+num)))

    a=[]
    xf,ts,x,y = cload(p)
    x  = x[-validation:]
    ts = ts[-validation:]
    l = m.predict(x)
    for i in range(len(ts)):
        dt=ts[i].split(" ")
        a.append(ind.indicator(p,"lookahead",tf,0,dt[0],dt[1],0))

    #r = open("combi_"+str(tf)+"/"+p+'.thresholds','r')
    #d = list(csv.reader(r, quoting=csv.QUOTE_NONNUMERIC))
    #r.close()
    #bc=d[1][0]
    #bcc=d[1][1]
    #sc=d[1][2]
    #scc=d[1][3]
    #bc2=d[1][4]
    #bcc2=d[1][5]
    #sc2=d[1][6]
    #scc2=d[1][7]
    #bc3=d[1][8]
    #bcc3=d[1][9]
    #sc3=d[1][10]
    #scc3=d[1][11]
    #print(p+":"+str(bc))
    #balance,m,wRate,cnt,pp,rpp,B = simulate(y,a,p, bc,bcc,sc,scc,  bc2,bcc2,sc2,scc2, bc3,bcc3,sc3,scc3)
    balance,m,wRate,cnt,pp,rpp,B = simulate(l,a,p, 9,0,0,0,  9,0,0,0,  9,0,0,0)
    fig = plt.figure()
    plt.plot(np.array(range(len(B))), np.array(B), label= '3x')
    plt.savefig("combi_"+str(tf)+"/"+p+"_testing_balance.png")
    plt.close()    
    m=d[0][1]
    print(p+": TEST : "+str(round(balance*m))+"  "+str(round(m,3))+ " lots")
    add_sum(balance*m)

def get_l1_models():
    models = {}
    #d = "combi_"+str(tf)+"/gradient_boosting_l1/"
    #d = "combi_"+str(tf)+"/xg_boosting_l1/"
    for c in ["EUR","GBP","AUD","CAD","CHF","NZD"]:
        acc = []
        for j in range(iterations):
            #if c=="AUD" and j==11: continue
            try:
                r = open(d+c+"."+str(j)+".regressor.accuracy",'r')
                acc.append(float(r.readline()))
                r.close()
            except:
                break
        m = max(acc)
        mi= acc.index(m)
        r = open(d+c+"."+str(mi)+".regressor",'rb')
        regressor = pickle.load(r)
        r.close()
        models[c] = regressor
        print(c+": "+str(m)+" ... model"+str(mi)+" chosen")
    return models

def load_l1_data(pairs,s):
    data = [{},{},{},{},{},{},{},{},{},{},{},{},{}]
    for i in range(13):
        p = pairs[i]
        if test!="": xfo,yf,ts,x,ya = pload("data/"+p+str(tf)+".s"+str(s))
        xf,yf,ts,x,yo = pload("data/"+p+str(tf)+test+".s"+str(s))
        for j in range(len(ts)): 
            if test!="": 
                for k in range(len(x[0])): x[j][k] = max(-1,min(1,x[j][k]*xf[k]/xfo[k]))
            data[i][ts[j]]=[x[j],yo[j]]
        print(pairs[0]+": "+p+"("+str(i)+")...loaded")
    return data

def get_correlated_pairs(p):
    results = []
    pre1 = p[:3]
    suf1 = p[3:]
    for pair in pairs99:
        if p==pair:
            results = [pair] + results
        else:
            pre2 = pair[:3]
            suf2 = pair[3:]
            if (pre1==pre2 or pre1==suf2 or suf1==pre2 or suf1==suf2): results.append(pair)
    return results

def get_best_l2_model(p):
    csvfile = open("combi_"+str(tf)+"/"+p+str(tf)+'.results','r', newline='')
    le = list(csv.reader(csvfile))
    csvfile.close()
    maxE=-999999999
    for i in range(len(le)-1,-1,-1):
        if len(le[i][0].split(':'))<11: continue
        e = float(le[i][0].split(':')[6]) 
        if e>maxE:
            maxE=e
            mi = int(le[i][0].split(':')[3]) 
            ml = float(le[i][0].split(':')[5]) 
            en = float(le[i][0].split(':')[7]) 
            ex = float(le[i][0].split(':')[8]) 
            sh = int(le[i][0].split(':')[10]) 
    print(p+": model "+str(mi)+" chosen")
    r = open("combi_"+str(tf)+"/gradient_boosting_l2/"+p+"_"+str(mi)+"_"+str(sh)+".o.regressor",'rb')
    regressor = pickle.load(r)
    r.close()
    print("level 2: "+str(mi)+"_"+str(sh)+" chosen")
    return regressor,ml,maxE,mi,en,ex,sh

def record(p,x1,pa):
    y,m,d,h,n,s = today_datetime()
    dt = y*10000 + m*100 + d
    w = open(p+str(tf)+'.records', 'a')
    w.write(str(dt)+',"'+str(time.localtime().tm_hour)+":"+str(time.localtime().tm_min)+":"+str(time.localtime().tm_sec)+" >> "+'"'+","+str(round(pa,5))+"\n")
    w.write(','.join(map(str,x1))+"\n")
    w.close()

def return_metatrader_data(p,d,h):
    csvfile = open(p+str(tf)+'.records', 'r',newline='')
    lines = list(csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC))
    csvfile.close()
    h = h+8
    if h>=24:
        h=h%24
        d=d+1
    for i in range(0,len(lines),3):
        if int(lines[i][0])+100==d and int(lines[i][1].split(':')[0])==h and int(lines[i][1].split(':')[1])==0:
            return lines[i+1],lines[i+2],float(lines[i][2]),lines[i+4],lines[i+5],float(lines[i+3][2]),float(lines[i][3]),float(lines[i][4])

def return_backtest_data(pair,d,h):
    pairs = get_correlated_pairs(pair)
    x1  = []
    x11 = []
    for p in pairs:
        xfo,yfo,tso,x,y = pload("data/"+p+str(tf)+".s0")
        xft,yft,tst,x,y = pload("data/"+p+str(tf)+test+".s0")
        x1l = len(xft)
        for j in range(len(x)): 
            if tst[j]==str(d)+" "+str(h).zfill(2)+":00:00":
                for i in range(x1l): x1.append(max(-1,min(1,x[j][i]*xft[i]/xfo[i])))
            if tst[j]==str(d)+" "+str(h).zfill(2)+":30:00":
                for i in range(x1l): x11.append(max(-1,min(1,x[j][i]*xft[i]/xfo[i])))                
    x2  = []
    x22 = []
    models = get_l1_models(pairs)            # models:[13][2*5*3]
    print("l1 models loaded")
    for i in range(13):
        for m in models[i]:
            x2.append(m.predict([x1[i*x1l:(i+1)*x1l]])[0])
            x22.append(m.predict([x11[i*x1l:(i+1)*x1l]])[0])
    xfo,yf,ts,x,y = pload("combi_"+str(tf)+"/l1_merger/"+pair+"_0.merger.o2")
    for i in range(len(x2)): x2[i] = max(-1,min(1,x2[i]/xfo[i]))
    m2a,lots,earning,mi,en,ex,sh = get_best_l2_model(pair)
    print("l2 model loaded")
    pa = m2a.predict([x2])[0]
    paa= m2a.predict([x22])[0]

    return x1,x2,pa,x11,x22,paa

def empty(f):
    return (not Path(f).is_file()) or Path(f).stat().st_size==0

if __name__ == '__main__':
    pl=[]
    #initialize_last_close(pairs99)
    if argv[1]=="test_i":
        ind.test("CADCHF",60,10,20190903,"07:00:00")
    elif argv[1]=="degap":
        pool = mp.Pool(28)
        for p in pairs99: pl.append(pool.apply_async(degap, (p,tf)))
        for pl2 in pl: pl2.get()
    elif argv[1]=="prepare":
        pool = mp.Pool(14)
        for p in pairs99 + chosen: pl.append(pool.apply_async(prepare,(p,tf)))
        for pl2 in pl: pl2.get()
    elif argv[1]=="combi":
        combi(tf)
    elif argv[1]=="train":
        combi(tf)
        #d = "combi_"+str(tf)+"/neural_network"
        #d = "combi_"+str(tf)+"/svr"
        d = "combi_"+str(tf)+"/gradient_boosting_l1"
        #d = "combi_"+str(tf)+"/xg_boosting_l1"
        pool = mp.Pool(13)
        for i in range(iterations):
            for p in pairs99:
                if empty(d+"/"+p+"."+str(i)+".regressor"):
                    if d=="combi_"+str(tf)+"/gradient_boosting_l1":
                        pl.append(pool.apply_async(gb_train,(p, d, p+"."+str(i)+".regressor")))
                    if d=="combi_"+str(tf)+"/xg_boosting_l1":
                        pl.append(pool.apply_async(xb_train,(p, d, p+"."+str(i)+".regressor")))
                    if d=="combi_"+str(tf)+"/neural_network":
                        pl.append(pool.apply_async(mlp_train,(p, d, p+"."+str(i)+".regressor")))
                    if d=="combi_"+str(tf)+"/svr":
                        pl.append(pool.apply_async(svr_train,(p, d, p+"."+str(i)+".regressor")))                        
            break
        for pl2 in pl: pl2.get()
    elif argv[1]=="simulate":
        simulate()
    elif argv[1]=="prepare_realtime":
        models = {}
        xf = {}
        models = get_l1_models()        
        combi(tf)

        for p in pairs99:
            print("storing "+p)
            xf[p],ts,x,y = cload(p)
        w = open("combi_"+str(tf)+"/realtime.factors",'wb')
        pickle.dump([models,xf],w,protocol=pickle.HIGHEST_PROTOCOL)
        w.close()
    elif argv[1]=="obtain_threshold":
        combi(tf)
        r = open("combi_"+str(tf)+"/realtime.factors",'rb')
        d = pickle.load(r)
        r.close()
        models = d[0]
        xf = d[1]
        models["USD"] = {}
        print("models and factors loaded")        
        pool = mp.Pool(1)
        for p in pairs99 + chosen: 
            if "JPY" in p:continue
            pre = p[:3]
            suf = p[-3:]
            pl.append(pool.apply_async(obtain_threshold,(pre,suf,models[pre],models[suf])))
        for pl2 in pl: pl2.get()
    elif argv[1]=="squeeze_threshold":
        combi(tf)
        r = open("combi_"+str(tf)+"/realtime.factors",'rb')
        d = pickle.load(r)
        r.close()
        models = d[0]
        xf = d[1]
        models["USD"] = {}
        print("models and factors loaded")        
        pool = mp.Pool(1)
        for p in pairs99 + chosen: 
            if "JPY" in p:continue
            pre = p[:3]
            suf = p[-3:]
            pl.append(pool.apply_async(squeeze_threshold,(pre,suf,models[pre],models[suf],int(argv[2]))))
        for pl2 in pl: pl2.get()        
    elif argv[1]=="test":
        combi(tf)
        reset_sum()
        r = open("combi_"+str(tf)+"/realtime.factors",'rb')
        d = pickle.load(r)
        r.close()
        models = d[0]
        xf = d[1]
        models["USD"] = {}
        print("models and factors loaded")        
        pool = mp.Pool(1)
        for p in pairs99 + chosen: 
            pl.append(pool.apply_async(test,(p,models[p])))
        for pl2 in pl: pl2.get()            
    elif argv[1]=="realtime":
        r = open("combi_"+str(tf)+"/realtime.factors",'rb')
        d = pickle.load(r)
        r.close()
        models = d[0]
        xf = d[1]
        print("models and factors loaded")

        def st(num):
            return str(round((num-1)*10000000))

        def predict(cc):
            csvfile = open('porting/combi'+str(tf)+'.export', 'r',newline='')
            lt = list(csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC))
            csvfile.close()
            x1 = lt[-1]
            for i in range(len(x1)): 
                if cc=="USD": 
                    pa=1
                else:
                    x1[i] = max(-1,min(1,x1[i]/xf[cc][i]))
                    pa = models[cc].predict([x1])[0]
            print(cc+" "+str(time.localtime().tm_hour).zfill(2)+":"+str(time.localtime().tm_min).zfill(2)+":"+str(time.localtime().tm_sec).zfill(2)+" >> "+":" + st(pa))
            record(cc,x1,pa)
            return pa

        firstTime = True
        while True:
            time.sleep(3)
            if (not firstTime) and (time.localtime().tm_min%30>0 or time.localtime().tm_sec>5): continue
            firstTime = False
            print("-----------------------------------------------------------------------------------------------------------------------------------------------------------------")
            pt = {}       # potential
            for cc in curr: 
                if cc=="JPY": continue
                pt[cc]=predict(cc)
            w = open('/mnt/c/Users/Intel/AppData/Roaming/MetaQuotes/Terminal/Common/Files/porting/combi'+str(tf)+'.import', 'w',newline='')
            for p in pairs99:
                if "JPY" in p: continue
                pre=p[:3]
                suf=p[-3:]
                r = open("combi_"+str(tf)+"/"+p+'.thresholds','r')
                d = list(csv.reader(r, quoting=csv.QUOTE_NONNUMERIC))
                r.close()
                s=pt[pre]/pt[suf]    # strength
                m=d[0][1]
                bc=d[1][0]
                bcc=d[1][1]
                sc=d[1][2]
                scc=d[1][3]
                if s>bc   : 
                    w.write(p+","+str(m)+"\n")
                    print(p+","+str(m)+",\t"+st(s)+"..."+st(bc)+","+st(bcc)+","+st(sc)+","+st(scc))
                elif s>scc: 
                    w.write(p+",999\n")
                    print(p+",999,\t"+st(s)+"..."+st(bc)+","+st(bcc)+","+st(sc)+","+st(scc))
                elif s<sc : 
                    w.write(p+","+str(-m)+"\n")
                    print(p+","+str(-m)+",\t"+st(s)+"..."+st(bc)+","+st(bcc)+","+st(sc)+","+st(scc))
                elif s<bcc: 
                    w.write(p+",-999\n")
                    print(p+",-999,\t"+st(s)+"..."+st(bc)+","+st(bcc)+","+st(sc)+","+st(scc))
                else:
                    w.write(p+",0,"+str(round(s,8))+"\n")
                    print(p+",0,\t"+st(s)+"..."+st(bc)+","+st(bcc)+","+st(sc)+","+st(scc))
            w.close()

    elif argv[1]=="generateTF": generateTF(argv[2])

    elif argv[1]=="verify":    # degap -> prepare -> verify -> compare
        pr = argv[2]
        dt = 20191008
        hr = 5

        labels1 = []
        labels2 = []
        base = ['low','high','dif1','dif2','dif3','dif4','dif5','dif6','dif7','dif8','dif9','dif10','dif11','dif12']
        for i in range(13):
            labels1 = labels1 + base
            for j in [12,18,24,30,40,60,80,100,150,200,400,600,800,1000,1500,2000]:
                labels1.append(str(i)+"_"+str(j)+"_ema")
                labels1.append(str(i)+"_"+str(j)+"_ema_dif")
                labels1.append(str(i)+"_"+str(j)+"_wpr")
                labels1.append(str(i)+"_"+str(j)+"_sd")
                labels1.append(str(i)+"_"+str(j)+"_bb")
                labels1.append(str(i)+"_"+str(j)+"_breakout")
                labels1.append(str(i)+"_"+str(j)+"_cci")
                labels1.append(str(i)+"_"+str(j)+"_rsi")
                labels1.append(str(i)+"_"+str(j)+"_atr")
                labels1.append(str(i)+"_"+str(j)+"_demarker")
                labels1.append(str(i)+"_"+str(j)+"_momentum")
            labels1.append("hour")
        for p in get_correlated_pairs(pr):
            for i in range(12): labels2.append(p+"_"+str(i))
        x1a,x2a,paa,q,q,q,q,q = return_metatrader_data(pr,dt,hr)
        x1b,x2b,pab,q,q,q,    = return_backtest_data(pr,dt,hr)
        w = open('test_results.csv', 'w')
        for i in range(len(x1a)):
            if i%100==0: print("writing..."+str(i))
            if i==0      : w.write(str(paa)+",,"+str(pab)+",\n")
            if i<len(x2b): w.write(labels1[i]+","+str(x1a[i])+","+str(x2a[i])+","+str(x1b[i])+","+str(x2b[i])+","+labels2[i]+",,"+str(abs(x1b[i]-x1a[i]))+","+str(abs(x2b[i]-x2a[i]))+"\n")
            else         : w.write(labels1[i]+","+str(x1a[i])+",0,"+str(x1b[i])+",0,,,"+str(abs(x1b[i]-x1a[i]))+"\n")
        gsum = 0
        isum = {}
        psum = {}
        for i in range(190):
            sum = 0
            for j in range(i,2483,191):
                err = abs(x1b[j]-x1a[j])
                sum = sum + err
                gsum = gsum + err
                inx = str(int(j/191))
                if inx in psum: psum[inx] = psum[inx] + err
                else: psum[inx] = err
                for ind in ['ema','dif','wpr','sd','bb','breakout','cci','rsi','atr','demarker','momentum']:
                    if ind in labels1[j]:
                        if ind in isum: isum[ind] = isum[ind]+err
                        else: isum[ind]=err
            print(f'{labels1[i]:16}'[2:]+":"+str(round(sum/13*100,2))+"%")
        print("")
        p = get_correlated_pairs(pr)
        for i in range(13): print(p[i]+" error: "+str(round(psum[str(i)]/190*100,2))+"%")
        print("")
        for ind in ['ema','dif','wpr','sd','bb','breakout','cci','rsi','atr','demarker','momentum']: print(f'{ind:10}' +' error:' +str(round(isum[ind]/16/13*100,2))+'%')
        print("")
        print("Level 1 error: "+str(round(gsum/13/190*100,2))+"%")
        print("")
        gsum = 0
        for i in range(len(x2b)): gsum = gsum + abs(x2b[i]-x2a[i])
        print("Level 2 error: "+str(round(gsum/len(x2b)*100,2))+"%")
        w.close()
    
    elif argv[1]=="compare":
        pr = argv[2]
        paa=[0,0]
        pab=[0,0]
        tradeA = 0
        tradeB = 0
        w = open('compare_results.csv', 'w')
        for dt in range(20191007,20191011):
            for hr in range(24):
                x1a,x2a,paa[0],x1a,x2a,paa[1],en,ex = return_metatrader_data(pr,dt,hr)
                x1b,x2b,pab[0],x1b,x2b,pab[1]       = return_backtest_data(pr,dt,hr)

                if tradeA==1 and paa[0]<-ex: tradeA=0
                elif tradeA==-1 and paa[0]>ex: tradeA=0
                if paa[0]>en: tradeA = 1
                elif paa[0]<-en: tradeA = -1
                if tradeB==1 and pab[0]<-ex: tradeB=0
                elif tradeB==-1 and pab[0]>ex: tradeB=0
                if pab[0]>en: tradeB = 1
                elif pab[0]<-en: tradeB = -1
                w.write(str(dt)+","+str(hr)+":00,"+str(paa[0])+","+str(pab[0])+","+str(tradeA)+","+str(tradeB)+"\n")

                if tradeA==1 and paa[1]<-ex: tradeA=0
                elif tradeA==-1 and paa[1]>ex: tradeA=0
                if paa[1]>en: tradeA = 1
                elif paa[1]<-en: tradeA = -1
                if tradeB==1 and pab[1]<-ex: tradeB=0
                elif tradeB==-1 and pab[1]>ex: tradeB=0
                if pab[1]>en: tradeB = 1
                elif pab[1]<-en: tradeB = -1
                w.write(str(dt)+","+str(hr)+":30,"+str(paa[1])+","+str(pab[1])+","+str(tradeA)+","+str(tradeB)+"\n")
        w.close()

    elif argv[1]=="compare_training_lengths":
        d="test"
        def compare_tl(i):
            sum = 0
            for j in range(5):
                sum = sum + gb_train("data/AUDNZD30.s0", d, "EURUSD.regressor", i)
            print("-------------------------------------------------------------------------------"+str(i)+ ": AVERAGE ACCURACY: " + str(sum/5))
        pool = mp.Pool(20)
        for i in range(0,100,5):
            pl.append(pool.apply_async(compare_tl,[i]))
        for pl2 in pl: pl2.get()   