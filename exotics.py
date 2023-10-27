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
from sklearn.svm import LinearSVR
from sklearn.neural_network import MLPRegressor
#from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgboost
import indicators as ind
from symbols import *
from math import sqrt
import matplotlib.pyplot as plt

tf = 120
iterations = 1
boundary = 0.38
#starting   = int(60/tf*24*23*39)
validation = int(60/tf*24*23*2)
testing    = int(60/tf*24*23*2)
start = 20080101
#test = "_test"
test = ""

#d = "combi_"+str(tf)+"/neural_network"
#d = "combi_"+str(tf)+"/svr"
d = "combi_"+str(tf)+"/gradient_boosting"
#d = "combi_"+str(tf)+"/xg_boosting"

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
    r = open('data/'+p+str(tf)+'.degap.csv','r')
    r.readline()
    d=0
    while d<start: 
        d,t,o,h,l,c,a,v = ind.parseLine(r)
    w=[]
    x=[]
    yo=[]
    y=[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
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
        if ema!=-999 and ema2!=-999:        
            #xx.append(ind.indicator(p,"psar",tf,-1,d,t,1))
            #if (cnt==0): xfc.append(0)
            xx.append(float(t.split(":")[0]))
            if (cnt==0): xfc.append(24)
            x.append(xx)
            for shift in range(16): 
                if o==0: o=oo
                la = ind.indicator(p,"lookahead",tf,0,d,t,shift)/o
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
    for i in range(16): 
        finalize_training_data("data/"+p+str(tf)+".s"+str(i),xf,ts,x,y[i])
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
        #print(file+" loaded")
    return d[0],d[1],d[2],np.array(d[3]),np.array(d[4])

def accuracy(actual,predictions):
    hits = 0
    zeroes = 0
    sum = 0
    l = 0
    for i in range(len(actual)):
        if np.isnan(predictions[i]): continue
        l=l+1
        if (actual[i]>0 and predictions[i]>0) or (actual[i]<0 and predictions[i]<0): hits+=1
        if actual[i]*predictions[i]==0: zeroes+=1
        #sum = sum + (actual[i]*(predictions[i]/abs(predictions[i])))
    return round(hits/l,4),sum/l,round(zeroes/l,4)

def my_score(a,y):   # actuals, predictions
    r = 0
    v = 0
    for i in range(len(a)): 
        r = r + a[i] * y[i]
        v = v + abs(a[i]) * abs(y[i])
    return round(r / v * 100,1)

def gb_train(s,c):
    lr = 0.2
    best_score=-9999
    best_regressor=None
    LV = randint(50,500)
    SP = randint(2,13)
    SS = uniform(0.3,0.9)
    r=GradientBoostingRegressor(verbose=0, warm_start=True, loss='huber', learning_rate=lr, n_estimators=1,max_depth=13,min_samples_leaf=LV,min_samples_split=SP,subsample=SS)
    outfile = c+"."+str(s)+"_"+str(LV)+"_"+str(SP)+"_"+str(SS)+".regressor"
    print("training "+c+str(s)+"------------------->"+d+"/"+outfile)
    bj=0
    for j in range(1,3000):
        w = open(d+"/"+outfile+".tmp",'wb')
        pickle.dump([r,j,bj,best_score,best_regressor],w)
        w.close()
        r.set_params(n_estimators=j)
        r.fit(xx[s][c],yy[s][c])
        #score = r.score(xt,yt[s][c])
        #score = r2_score(yt[s][c],r.predict(xt))
        score = my_score(yt[s][c],r.predict(xt[s][c]))
        if score>best_score:
            bj=j
            best_score = score
            best_regressor = deepcopy(r)
            print(outfile+":"+str(j)+":  "+str(score)+" <<<")
        else:
            print(outfile+":"+str(j)+":  "+str(score)+"......"+str(r.get_params()["learning_rate"]))
        if j-bj>6: break
    Path(d+"/"+outfile+".tmp").unlink()
    if best_score>0:
        if not os.path.exists(d): os.makedirs(d)
        w = open(d+"/"+outfile,'wb')
        pickle.dump(best_regressor,w)
        w.close()
        w = open(d+"/"+outfile+".accuracy",'w')
        w.write(str(best_score))
        w.close()    
        print("written to "+d+"/"+outfile)

def resume_gb_train(s,c,outfile,r,j_start,bj,best_score,best_regressor):
    print("(RESUME) training "+c+str(s)+"------------------->"+d+"/"+outfile)
    for j in range(j_start,3000):
        w = open(d+"/"+outfile+".tmp",'wb')
        pickle.dump([r,j,bj,best_score,best_regressor],w)
        w.close()        
        r.set_params(n_estimators=j)
        r.fit(xx,yy[s][c])
        #score = r.score(xt,yt[s][c])
        score = r2_score(yt[s][c],r.predict(xt))
        if score>best_score:
            bj=j
            best_score = score
            best_regressor = deepcopy(r)
            print(outfile+":"+str(j)+":  "+str(score)+" <<<")
        else:
            print(outfile+":"+str(j)+":  "+str(score)+"......"+str(r.get_params()["learning_rate"]))
        if j-bj>6: break
    Path(d+"/"+outfile+".tmp").unlink()
    if best_score>0:
        if not os.path.exists(d): os.makedirs(d)
        w = open(d+"/"+outfile,'wb')
        pickle.dump(best_regressor,w)
        w.close()
        w = open(d+"/"+outfile+".accuracy",'w')
        w.write(str(best_score))
        w.close()    
        print("written to "+d+"/"+outfile)

def simulate(lot,a,pair):
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
    #print(lot[:10])
    for i in range(0,len(lot)):
        lt = lot[i]
        if ol>=0:
            if lt>ol: l=lt
            if lt<0: l=lt
        if ol<=0:
            if lt<ol: l=lt
            if lt>0: l=lt
        #l = lt
        if abs(l)>0.00000000001: tcnt+=1
        r=a[i]
        if (r>0 and l>0) or (r<0 and l<0):
            positive+=1
        if abs(l)>0.000000005:
            rpositive+=1
        profit = (l * r - abs(l - ol) * spread[pair] / 2) * unit[pair] / value[pair[-3:]] - abs(l - ol) * commission[pair] / 2
        #profit = (l * r) * unit[pair] / value[pair[-3:]] 
        if profit > 0: pcnt+=1
        if profit > 1200: profit=1200
        if profit <-1200: profit=-1200
        balance += profit
        if balance > maxB: maxB = balance
        if maxB - balance > maxDD: maxDD = maxB - balance
        B.append(balance)
        ol=l
    cnt = len(lot)
    #print(pair+":"+str(maxDD)+" "+str(cnt)+" "+str(tcnt))
    if maxDD==0 or cnt==0 or tcnt<5:
        return balance/1000,0.001,0,tcnt,0,0,B
    #if uniform(0,1)<0.001: 
    #print(pair+":"+str(balance)+":"+str(tcnt))        
    return balance, -100/-maxDD, float(pcnt)/cnt,tcnt,fdp(float(positive)/cnt),fdp(float(rpositive)/cnt),B

def reset_sum():
    w = open("combi_"+str(tf)+"/sum.txt",'w')
    w.write("0")
    w.close()

def test(p,m,ts,x,index):
    def add_sum(num):
        r = open("combi_"+str(tf)+"/sum.txt",'r')
        d = list(csv.reader(r, quoting=csv.QUOTE_NONNUMERIC))
        r.close()
        w = open("combi_"+str(tf)+"/sum.txt",'w')
        w.write(str(float(d[0][0])+num))
        w.close()
        print("SUM added: "+str(round(d[0][0]+num)))

    a = []
    lot = m.predict(x)
    for i in range(len(ts)):
        lot[i] = lot[i]
        dt=ts[i].split(" ")
        a.append(ind.indicator(p,"lookahead",tf,0,dt[0],dt[1],0))

    #r = open("combi_"+str(tf)+"/"+p+'.index','r')
    #d = list(csv.reader(r, quoting=csv.QUOTE_NONNUMERIC))
    #r.close()
    #print(p+":"+str(bc))
    balance,m,wRate,cnt,pp,rpp,B = simulate(lot,a,p)
    #balance,m,wRate,cnt,pp,rpp,B = simulate3(lot,a,p,d[0][0],d[0][1],d[0][2],d[0][3])
    fig = plt.figure()
    plt.plot(np.array(range(len(B))), np.array(B), label= '3x')
    plt.savefig("combi_"+str(tf)+"/"+p+"_final_testing_balance.png")
    plt.close()    
    #m=d[0][1]
    print(p+": TEST : "+str(round(balance*m))+"  "+str(round(m,3))+ " lots")
    #add_sum(balance*m)
    return balance * m, B, m, cnt    

def get_models():
    models = {}
    for p in exotics:
        acc = []
        files=glob(d+"/"+p+"*regressor.accuracy")
        mx = -999
        mf = ""
        ms = -1
        for f in files:
            r = open(f,'r')
            acc = float(r.readline())
            r.close()
            if acc>mx:
                if tf<100: ms=f[34:36]
                else: ms=f[35:37]
                if ms[-1]=="_": ms = int(ms[:-1])
                else          : ms = int(ms)
                if ms>18: continue
                mx=acc
                mf=f
        if mx>-99:
            print(mf+" chosen: "+str(ms)+" : "+str(mx))
            r = open(mf[:-9],'rb')
            models[p]=pickle.load(r)
            r.close()
    return models

def get_very_best_models():
    models = {}
    for c in exotics:
        mx = -999
        mf = ""
        ms = -1
        me = -999999
        mt = -1
        for t in [120,240,360,480]:
            xf,yf,ts,x,y = pload("data/"+c+str(t)+".s"+str(0))
            ts=ts[-testing:]
            x = x[-testing:]
            dd = "combi_"+str(t)+"/gradient_boosting"
            files=glob(dd+"/"+c+"*regressor.accuracy")
            for f in files:
                r = open(f,'r')
                acc = float(r.readline())
                r.close()
                r = open(f[:-9],'rb')
                mo = pickle.load(r)
                r.close()
                earning,B,m,cnt = test(c,mo,ts,x,0)
                print(f+": "+str(ms)+" : "+str(acc) + " : "+str(earning)+"("+str(cnt)+")")
                if earning*acc>me and earning<5000:
                    if t<100: ms=f[34:36]
                    else: ms=f[35:37]
                    if ms[-1]=="_": ms = int(ms[:-1])
                    else          : ms = int(ms)
                    if ms>18: continue
                    me=earning*acc
                    mx=acc
                    mf=f
                    mt=t
        if me>-99:
            r = open(mf[:-9],'rb')
            models[c]=pickle.load(r)
            r.close()
            t = int(mf[6:9])
            xf,yf,ts,x,y = pload("data/"+c+str(t)+".s"+str(0))
            ts=ts[-testing:]
            x = x[-testing:]
            earning,B,m,cnt = test(c,models[c],ts,x,0)
            fig = plt.figure()
            plt.plot(np.array(range(len(B))), np.array(B), label= '3x')
            plt.savefig("final_selection/"+c+".png")
            plt.close()
            w = open("final_selection/"+c+".lots",'w')
            w.write(str(m)+","+str(mt))
            w.close()                
            print(mf+" chosen: "+str(ms)+" : "+str(mx) + " : "+str(round(me/mx))+"("+str(cnt)+")\n\n\n\n")
    return models

def get_l1_models():
    models = []
    for c in ["JPY","EUR","GBP","AUD","CAD","CHF","NZD"]:
        acc = []
        for s in range(12):
            files=glob(d+"/"+c+"."+str(s)+"_*accuracy")
            mx = -999
            mf = ""
            ms = -1
            for f in files:
                r = open(f,'r')
                acc = float(r.readline())
                r.close()
                if acc>mx:
                    mx=acc
                    mf=f
                    if tf<100: ms=f[31:33]
                    else: ms=f[32:34]
                    if ms[-1]=="_": ms = int(ms[:-1])
                    else          : ms = int(ms)
            if mx>-99:
                print(mf+" chosen: "+str(ms)+" : "+str(mx))
                r = open(mf[:-9],'rb')
                models.append(pickle.load(r))
                r.close()
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

def record(p,x1,pa):
    y,m,d,h,n,s = today_datetime()
    dt = y*10000 + m*100 + d
    w = open(p+str(tf)+'.records', 'a')
    w.write(str(dt)+',"'+str(time.localtime().tm_hour)+":"+str(time.localtime().tm_min)+":"+str(time.localtime().tm_sec)+" >> "+'"'+","+str(round(pa,5))+"\n")
    w.write(','.join(map(str,x1))+"\n")
    w.close()

def empty(f):
    return (not Path(f).is_file()) or Path(f).stat().st_size==0

def save_validation():
    with gzip.open("data/combi"+str(tf)+".x2", 'wb') as f:
        ubjson.dump([xf,ts,x2,y[0]], f)

if __name__ == '__main__':
    pl=[]
    #initialize_last_close(pairs99)
    if argv[1]=="degap":
        for p in exotics:
            os.rename("Data/"+p+str(tf)+".csv","Data/"+p+str(tf)+".degap.csv")
            print("Data/"+p+str(tf)+".csv --> Data/"+p+str(tf)+".degap.csv")
    elif argv[1]=="prepare":
        pool = mp.Pool(14)
        #pairs99 = ["EURCAD","EURCHF","EURAUD","EURNZD","GBPCAD","GBPCHF","GBPAUD","GBPNZD","CADCHF","AUDCAD","NZDCAD","AUDCHF","NZDCHF","AUDNZD"]
        for p in exotics: 
            pl.append(pool.apply_async(prepare,(p,tf)))
        for pl2 in pl: pl2.get()
    elif argv[1]=="train":
        xx = [{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}]
        xt = [{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}]
        yy = [{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}]
        yt = [{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}]
        for shift in range(16):
            for p in exotics:
                xf,yf,ts,x,y = pload("data/"+p+str(tf)+".s"+str(shift))
                xx[shift][p], xt[shift][p], yy[shift][p], yt[shift][p] = train_test_split(x[:-validation],y[:-validation],test_size=0.15, random_state=1688, shuffle=False)
                #while (len(xx[0])!=5348): 
                #    xx = xx[1:]
                #    yy[shift][p] = yy[shift][p][1:]
                print(p+" "+str(shift)+": Train Test Split X_33333 Veritication: " + str(xx[shift][p][333][0])+"   Y:"+str(yy[shift][p][333]))
        pool = mp.Pool(28)
        for iter in range(iterations):
            for shift in range(16):
                for p in exotics:
                    pl.append(pool.apply_async(gb_train, (shift,p)))
        for pl2 in pl: pl2.get()
    elif argv[1]=="obtain_threshold":
        if True:
        #if False:
            models = get_models()
            xf,ts,x,y = load_combi()
        else:
            models = get_l2_models()
            xf,ts,x,y = load_combi("2")
        ts=ts[-validation:-testing]
        x = x[-validation:-testing]
        y = []
        pool = mp.Pool(7)
        for c in curr2:
            pl.append(pool.apply_async(obtain_threshold2,(c,models[c],ts,x)))
        for pl2 in pl: pl2.get()    
    elif argv[1]=="test":
        models = get_models()
        pool = mp.Pool(7)
        for p in exotics:
            xf,yf,ts,x,y = pload("data/"+p+str(tf)+".s"+str(0))
            ts = ts[-testing:]
            x = x[-testing:]
            pl.append(pool.apply_async(test,(p,models[p],ts,x,1)))
        for pl2 in pl: pl2.get()     
    elif argv[1]=="prepare_realtime":
        models=get_very_best_models()
        xf = {}
        for tff in [120,240,360,480]:
            xf[tff] = {}
            for p in exotics:
                xf[tff][p],yf,ts,x,y = pload("data/"+p+str(tff)+".s"+str(0))
        lots = {}
        tfm = {}
        for p in exotics:
            r = open("final_selection/"+p+".lots",'r')
            d = list(csv.reader(r, quoting=csv.QUOTE_NONNUMERIC))
            r.close()
            lots[p] = float(d[0][0])
            tfm[p] = int(d[0][1])
        print("models and factors loaded")
        with gzip.open("final_selection/exotics.realtime", 'wb') as f:
            pickle.dump([models,xf,lots,tfm], f)
    elif argv[1]=="realtime":
        with gzip.open("final_selection/exotics.realtime", 'rb') as f:
            data=pickle.load(f)   
        models=data[0]
        xf=data[1]
        lots=data[2]
        tfm=data[3]

        def st(num):
            return str(round(num*100000))

        def predict(p):
            csvfile = open('porting/'+p+'.export', 'r',newline='')
            lt = list(csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC))
            csvfile.close()
            x1 = lt[-1]
            for i in range(len(x1)): 
                try:
                    x1[i] = max(-1,min(1,x1[i]/xf[tfm[p]][p][i]))
                except:
                    x1[i] = 0
            pa = models[p].predict([x1])[0] * lots[p]
            #print(p+" "+str(time.localtime().tm_hour).zfill(2)+":"+str(time.localtime().tm_min).zfill(2)+":"+str(time.localtime().tm_sec).zfill(2)+" >> "+":" + str(pa))
            record(p,x1,pa)
            return pa

        firstTime = True
        while True:
            time.sleep(3)
            if (not firstTime) and (time.localtime().tm_min%15>0 or time.localtime().tm_sec>5): continue
            firstTime = False
            print("-----------------------------------------------------------------------------------------------------------------------------------------------------------------")
            for p in exotics:
                w = open('/mnt/c/Users/Intel/AppData/Roaming/MetaQuotes/Terminal/Common/Files/porting/'+p+'.import', 'w',newline='')
                pr = predict(p)
                w.write(str(pr)+"\n")
                print(p+" "+str(time.localtime().tm_hour).zfill(2)+":"+str(time.localtime().tm_min).zfill(2)+":"+str(time.localtime().tm_sec).zfill(2)+" >> "+str(round(pr,3)))
                w.close()

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