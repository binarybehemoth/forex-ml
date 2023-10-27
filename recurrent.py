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
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
import indicators as ind
from symbols import *
from math import sqrt
import matplotlib.pyplot as plt

mtt = 7    # months to test
tf = 120
iterations = 88
boundary = 0.38
#starting   = int(60/tf*24*23*39)
validation = int(60/tf*24*23*mtt)
#validation = int(60/tf*24*23*mtt*2)
testing    = 0#int(60/tf*24*23*mtt)
start = 20100101
#test = "_test"
test = ""

#d = "combi_"+str(tf)+"/svr"
#d = "combi_"+str(tf)+"/gradient_boosting"
d = "combi_"+str(tf)+"/rnn"
#d = "combi_"+str(tf)+"/neural____network"
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
        #if fac!=0:
        #    for j in range(l): x[j][i] = max(-1,min(1,x[j][i]/fac))
        #elif (fac>450): print("index "+str(i)+" exceeded 450................................")
        fac = 1
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
                la = ind.indicator(p,"lookahead",tf,0,d,t,shift)
                if la==-999:
                    print(p+" : invalid value at "+str(d)+" "+t+"..."+str(y[shift][-1]))
                    y[shift].append(0)
                else:
                    #la = la / o
                    y[shift].append(la)
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
        print(file+" loaded")
    return d[0],d[1],d[2],np.array(d[3]),np.array(d[4])

def combi(tf):
    global cache

    if not empty("data/combi"+str(tf)+".s"):
        r = open("data/combi"+str(tf)+".s","rb")
        cache["data/combi"+str(tf)+".s"] = pickle.load(r)
        r.close()
        return

    def pload2(file):
        r = open(file,'rb')
        d = pickle.load(r)
        r.close()
        print(file+" loaded")
        return d[0],d[1],d[2],np.array(d[3]),np.array(d[4])

    def get_change(yy):
        pc = [{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}]
        data = {}
        for p in pairs99:
            #if "JPY" in p: continue
            print("processing "+p)
            da = [{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}]
            for j in range(16):
                for i in range(len(ts[p])): 
                    da[j][ts[p][i]] = yy[p][j][i]
            data[p] = deepcopy(da)
        print("PROCESSED")
        for t in ts["EURUSD"]:
            for i in range(16):
                cs = {}  # currency strength
                all_in = True
                for p in pairs99:
                    #if "JPY" in p: continue
                    if not (t in data[p][i]):
                        all_in = False
                        break
                if (all_in):
                    pc[i][t] = {}
                    for p in pairs99:
                        #if "JPY" in p: continue
                        pc[i][t][p] = deepcopy(data[p][i][t])
        return pc

    xf = {}
    yf = {}
    ts = {}
    x = {}
    y = {}
    fp = {}
    data = {}
    for p in pairs99 + chosen:
        y[p] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        for i in range(16):xf[p],yf[p],ts[p],x[p],y[p][i] = pload2("data/"+p+str(tf)+".s"+str(i))
        #xf[p],yf[p],ts[p],x[p],y = pload2("data/"+p+str(tf)+".s0")
        print(str(len(ts[p]))+" records")
        data[p] = {}
        for i in range(len(ts[p])): data[p][ts[p][i]] = x[p][i]
        fp[p] = open('data/'+str(p)+str(tf)+'.degap.csv','r')
    print("saving...")
    w = open("data/combi"+str(tf)+".y","wb")
    pickle.dump(y,w,protocol=pickle.HIGHEST_PROTOCOL)
    w.close()
    #r = open("data/combi"+str(tf)+".y","rb")
    #y=pickle.load(r)
    #r.close()        
    c = get_change(y)
    print(str(len(c[0]))+" y records")
    y = None
    print("RESTRUCTURED")
    rxf = []
    rts = []
    rx  = []
    ry  = [{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}]
    for i in range(16):
        for p in pairs99: 
            #if "JPY" in p: continue
            ry[i][p] = []
    for p in pairs99 + chosen: rxf = rxf + xf[p]
    for t in c[0]:
        if not (t in c[0]):continue
        all_present = True
        for p in pairs99 + chosen:
            #if "JPY" in p: continue
            if not (t in data[p]):
                all_present = False
                print(p+": "+t+" not present")
                break
        if not all_present: continue
        if len(rx)%1000==0:print(t)
        xx  = []
        for p in pairs99 + chosen:xx = list(xx) + list(data[p][t])
        #for cc in curr2:
            #if cc=="JPY":continue
            #v = ind.indicator("combi","strength",tf,0,int(t.split()[0]),t.split()[1],1)
            #print(v)
            #print(int(t.split()[0]))
            #print(t.split()[1])
            #if v==-999: xx.append(0)
            #else: xx.append(v)
            #if (len(rx)==0):rxf.append(1)
        rts.append(t)
        rx.append(xx)
        for p in pairs99:
            #if "JPY" in p: continue
            for i in range(16): ry[i][p].append(c[i][t][p])
    print(str(len(rxf))+" features")
    print(str(len(rx[0]))+" features")
    print(str(len(rx))+" x records")
    print("ended at..."+rts[-1])
    print("Combi finalized")
    return rxf,rts,rx,ry


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

def rnn_train(p):
    outfile = p+".regressor"

    print("building model")
    regressor = Sequential()
    regressor.add(LSTM(units=50,return_sequences=True,input_shape=(xx.shape[1], xx.shape[2])))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units=50,return_sequences=True))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units=50,return_sequences=True))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units=50))
    regressor.add(Dropout(0.2))
    regressor.add(Dense(units=1))

    print("training "+p+"------------------->"+d+"/"+outfile)
    regressor.compile(optimizer='adam',loss='mean_squared_error')
    regressor.fit(xx,yy[p],epochs=6,batch_size=32,verbose=2)
    outfile = p+"."+".regressor"
    best_score,bm,bt,cnt,pp,rpp,B = simulate(regressor.predict(xt),a[p][ts:],p)
    if not os.path.exists(d): os.makedirs(d)
    print(outfile+":"+str(best_score)+":  "+str(bt)+":  "+str(bm)+" <<<")
    plt.plot(np.array(range(len(B))), np.array(B), label= '3x')
    plt.savefig(d+"/"+outfile+"_testing_balance.png")
    plt.close()          
    w = open(d+"/"+outfile,'wb')
    pickle.dump(regressor,w)
    w.close()
    w = open(d+"/"+outfile+".accuracy",'w')
    w.write(str(best_score)+","+str(bm))
    w.close()    
    print("written to "+d+"/"+outfile)

def simulate(lot,a,pair):
    sign = 1
    cnt = len(lot)
    for repeat in range(2):
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
        #print(lot[:10])
        for i in range(len(lot)):
            lt = lot[i]
            if lt>0  : l=sign
            elif lt<0: l=-sign
            else     : l=0
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
            balance += profit
            if balance > maxB: maxB = balance
            if maxB - balance > maxDD: maxDD = maxB - balance
            B.append(balance)
            ol=l
        if balance>0 or sign<0: break
        psave = [balance, (-100/-maxDD)*sign, float(pcnt)/cnt,tcnt,fdp(float(positive)/cnt),fdp(float(rpositive)/cnt),B]
        sign = -1
    #print(pair+":"+str(maxDD)+" "+str(cnt)+" "+str(tcnt))
    if maxDD==0 or cnt==0 or tcnt<5:
        return balance/1000,0.001,0,tcnt,0,0,B
    #if uniform(0,1)<0.001: 
    #print(pair+":"+str(balance)+":"+str(tcnt))
    n = len(B)
    
    #tg = 0     # total gain
    #tl = 0     # total loss
    #for i in range(n-1):
    #    if B[i+1]>B[i]: tg += B[i+1]-B[i]
    #    if B[i+1]<B[i]: tl += B[i]-B[i+1]
    #pf = tg/tl   # profit factor

    #mg = (B[-1] - B[0]) / (len(B)-1)
    #s = 0 # straightness
    #for i in range(n): s += abs(mg*i+B[0]-B[i])
    #for i in range(n-1): s += abs(B[i+1]-B[i]-mg)
    #for i in range(len((B))): B[i] = B[i] * (-100/-maxDD)
    #return balance*(-100/-maxDD), (-100/-maxDD), float(pcnt)/cnt,tcnt,fdp(float(positive)/cnt),fdp(float(rpositive)/cnt),B
    if sign>0: return balance, (-100/-maxDD)*sign, float(pcnt)/cnt,tcnt,fdp(float(positive)/cnt),fdp(float(rpositive)/cnt),B
    else:
        if psave[0]*psave[1] > balance * (-100/-maxDD): return psave[0],psave[1],psave[2],psave[3],psave[4],psave[5],psave[6]
        else                                          : return balance, (-100/-maxDD)*sign, float(pcnt)/cnt,tcnt,fdp(float(positive)/cnt),fdp(float(rpositive)/cnt),B

def save_combi():
    xf,ts,x,y = combi(tf)
    with gzip.open("data/combi"+str(tf)+".x", 'wb') as f:
        ubjson.dump([xf,ts,x,y], f)

def load_combi(i="",tm=tf):
    with gzip.open("data/combi"+str(tm)+".x"+i, 'rb') as f:
        data=ubjson.load(f)   
    return data[0],data[1],data[2],data[3]

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
        #pairs99 = ["EURCAD","EURCHF","EURAUD","EURNZD","GBPCAD","GBPCHF","GBPAUD","GBPNZD","CADCHF","AUDCAD","NZDCAD","AUDCHF","NZDCHF","AUDNZD"]
        for p in pairs99 + chosen: 
            pl.append(pool.apply_async(prepare,(p,tf)))
        for pl2 in pl: pl2.get()
    elif argv[1]=="combi":
        save_combi()
    elif argv[1]=="train":
        #save_combi()
        xf,ts,x,y = load_combi()
        print(str(len(y[0]["EURUSD"]))+" records")
        yy = {}
        yt = {}

        a = {}
        for p in pairs99:
            if "JPY" in p: continue
            a[p] = []
            for i in range(len(ts[-validation:])):
                dt=ts[-validation:][i].split(" ")
                a[p].append(ind.indicator(p,"lookahead",tf,0,dt[0],dt[1],0))

        xx = x[:-validation]
        xt = x[-validation:]
        scaler = MinMaxScaler()
        #print("Scaling")
        #xx_raw = scaler.fit_transform(xx)
        #xt_raw = scaler.fit_transform(xt)
        print("yy_raw yt_raw")
        for shift in range(16):
            for p in pairs99:
                if "JPY" in p: continue
                yy[p] = y[0][p][:-validation]
                yt[p] = y[0][p][-validation:]
        xx={}
        xt={}
        ts = 200
        #print("Preparing xx")
        #for i in range(ts, len(xx_raw)): xx.append(xx_raw[i-ts:i])
        #print("Preparing xt")
        #for i in range(ts, len(xt_raw)): xt.append(xt_raw[i-ts:i])
        print("Preparing yy,yt")
        xx = []
        for i in range(ts, len(x[:-validation])): 
            zz = []
            for j in range(i-ts,i):
                z = []
                for p in pairs99: 
                    if "JPY" in p: continue
                    z.append(yy[p][j])
                zz.append(z)
            xx.append(zz)     
        xx = np.array(xx)
        xt = []
        for i in range(ts, len(x[-validation:])): 
            zz = []
            for j in range(i-ts,i):
                z = []
                for p in pairs99: 
                    if "JPY" in p: continue
                    z.append(yt[p][j])
                zz.append(z)
            xt.append(zz)     
        xt = np.array(xt)
        for p in pairs99:
            if "JPY" in p: continue
            yy[p] = np.array(yy[p][ts:])
            yt[p] = np.array(yt[p][ts:])
        x=[]
        y=[]

        pool = mp.Pool(7)
        for iter in range(iterations):
            for p in pairs99:
                if "JPY" in p: continue
                pl.append(pool.apply_async(rnn_train, (p,)))
        for pl2 in pl: pl2.get()
    elif argv[1]=="prepare_realtime":
        models,bms,score = get_models()
        print("models and factors loaded")
        with gzip.open("combi.realtime", 'wb') as f:
            pickle.dump([models,bms,score], f)
    elif argv[1]=="realtime":
        with gzip.open("combi.realtime", 'rb') as f:
            data=pickle.load(f)   
        models= data[0]
        bms   = data[1]
        score = data[2]

        def st(num):
            return str(round(num*100000))

        def predict(s,p):
            pa = models[s][p].predict([x1])[0] 
            #print(cc+" "+str(time.localtime().tm_hour).zfill(2)+":"+str(time.localtime().tm_min).zfill(2)+":"+str(time.localtime().tm_sec).zfill(2)+" >> "+":" + str(pa))
            #record(p,x1,pa)
            return pa

        firstTime = True
        while True:
            time.sleep(3)
            if (not firstTime) and (time.localtime().tm_min>0 or time.localtime().tm_sec>2): continue
            firstTime = False
            print("-----------------------------------------------------------------------------------------------------------------------------------------------------------------")
            
            csvfile = open('porting/combi'+str(tf)+'.export', 'r',newline='')
            lt = list(csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC))
            csvfile.close()
            x1 = lt[-1]
            #for p  in ["GBPAUD","GBPUSD","AUDNZD"]:
            acc = {"USD":0, "EUR":0, "GBP":0, "AUD":0, "NZD":0, "CAD":0, "CHF":0}
            for s in range(16):
                for p in pairs99:
                    if "JPY" in p: continue
                    if not (p in models[s]): continue
                    pa = predict(s,p)
                    acc[p[:3]] += (pa/abs(pa)) * (bms[s][p]/abs(bms[s][p])) * score[s][p]
                    acc[p[3:]] -= (pa/abs(pa)) * (bms[s][p]/abs(bms[s][p])) * score[s][p]
            print(acc)
            maxK = max(acc, key=acc.get)
            minK = min(acc, key=acc.get)
            if (maxK+minK) in pairs99: 
                bestP=maxK+minK
                decision=1
            else: 
                bestP=minK+maxK
                decision=-1
            w = open('/mnt/c/Users/Intel/AppData/Roaming/MetaQuotes/Terminal/Common/Files/porting/combi120.import', 'w',newline='')
            w.write(bestP+","+str(decision))
            w.close()
            print(str(time.localtime().tm_hour).zfill(2)+":"+str(time.localtime().tm_min).zfill(2)+":"+str(time.localtime().tm_sec).zfill(2)+" >> \t"+str(decision)+" >> \t"+" ***\t"+bestP)
