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
from sklearn.svm import LinearSVR
from sklearn.neural_network import MLPRegressor
#from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgboost
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
d = "combi_"+str(tf)+"/random_forest"
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
    for i in range(800):
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
        for i in [1,2,3,4,5,6,7,8,9,10,25,50,100,200,400,800]:
            xx.append(o-w[800-i])
            if (cnt==0): xfc.append(0)
        w = w[1:]
        w.append(a)
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

def rf_train(s,p):
    lr = 0.03
    best_score=-9999999
    best_regressor=None
    LV = randint(50,500)
    SP = randint(2,13)
    SS = uniform(0.3,0.9)
    r = RandomForestRegressor(warm_start=True,verbose=2,n_jobs=34)
    outfile = p+"."+str(s)+"_"+str(LV)+"_"+str(SP)+"_"+str(SS)+".regressor"
    print("training "+p+str(s)+"------------------->"+d+"/"+outfile)
    bj=0
    bm=0
    for j in range(100,101,30):
        r.set_params(n_estimators=j)
        r.fit(xx,yy[s][p])
        score,m,bt,cnt,pp,rpp,B = simulate(r.predict(xt),a[p],p)
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
        if j-bj>5: break
    #Path(d+"/"+outfile+".tmp").unlink()
    #if best_score>0:
    if not os.path.exists(d): os.makedirs(d)
    w = open(d+"/"+outfile,'wb')
    pickle.dump(best_regressor,w)
    w.close()
    w = open(d+"/"+outfile+".accuracy",'w')
    w.write(str(best_score)+","+str(bm))
    w.close()    
    print("written to "+d+"/"+outfile)

def simulate(lot,a,pair):
    sign = 1
    cnt = len(lot)
    for repeat in range(1):
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
    if sign>0 or True: return balance, (-100/-maxDD)*sign, float(pcnt)/cnt,tcnt,fdp(float(positive)/cnt),fdp(float(rpositive)/cnt),B
    else:
        if psave[0]*psave[1] > balance * (-100/-maxDD): return psave[0],psave[1],psave[2],psave[3],psave[4],psave[5],psave[6]
        else                                          : return balance, (-100/-maxDD)*sign, float(pcnt)/cnt,tcnt,fdp(float(positive)/cnt),fdp(float(rpositive)/cnt),B

def obtain_threshold(c,m,ts,x):
    a = []
    lots = m.predict(x)
    if (c+"USD") in pairs99: p = c+"USD"
    else:                    p = "USD"+c    
    for i in range(len(ts)):
        dt=ts[i].split(" ")
        a.append(ind.indicator(p,"lookahead",tf,0,dt[0],dt[1],0))
    mx = -999999999
    bf = -1
    bi = -1
    bm = -1
    bB = None
    print("ACCURACY: "+str(accuracy(a,lots)))
    for k in range(100,500,10):
        for j in range(2000,10000,10):
            lot = deepcopy(lots)
            index = j / 100
            for i in range(len(ts)):
                l = lot[i]*k
                if (c+"USD") in pairs99: lot[i] =  (abs(l)/l)*(abs(l)**index)
                else:                    lot[i] = -(abs(l)/l)*(abs(l)**index)
            balance,m,wRate,cnt,pp,rpp,B = simulate(lot,a,p)
            #print(p+":"+str(k)+":"+str(index)+":"+str(round(balance*m,2))+":"+str(cnt)+":"+str(round(m,3)))
            if balance*m>mx:
                mx = balance*m
                bf = k
                bi = index
                bm = m
                bB = B
    print(p+" best index: "+str(bf)+" : "+str(bi)+" : "+str(round(mx)))
    w = open("combi_"+str(tf)+"/"+p+".index",'w')
    w.write(str(bf)+","+str(bi)+","+str(bm)+","+str(round(mx)))
    w.close()
    plt.plot(np.array(range(len(bB))), np.array(bB), label= '3x')
    plt.savefig("combi_"+str(tf)+"/"+p+"_testing_balance.png")
    plt.close()        

def obtain_threshold3(p,m,ts,x):
    a = []
    for i in range(len(ts)):
        dt=ts[i].split(" ")
        a.append(ind.indicator(p,"lookahead",tf,0,dt[0],dt[1],0))
    lots = m[p].predict(x)
    mx = -999999999
    bB = None
    bm = 0
    debug = False
    factor = 1000000
    for j in range(1000):
        t = 0.1 + float(j)/100.0
        pf,m,wRate,cnt,pp,rpp,B = simulate4(lots,a,p,t)
        if (debug): print(p+":"+str(t)+":   "+str(round(pf,4))+":"+str(cnt)+":"+str(round(m,3)))
        if cnt<0:break
        if pf>mx:
            mx = pf
            bt = t
            bB = B
            bm = m
    plt.plot(np.array(range(len(bB))), np.array(bB), label= '3x')
    plt.savefig("combi_"+str(tf)+"/"+p+"_testing_balance.png")
    plt.close()                                            
    #print(p+" best tresholds: "+str(round(mx))+"  :  \t"+str(bc)+"\t"+str(bcc)+"\t"+str(sc)+"\t"+str(scc))
    print(p+" best thresholds: "+str(round(mx,5))+"  :  \t"+str(bt))
    w = open("combi_"+str(tf)+"/"+p+".index",'w')
    w.write(str(bt)+","+str(round(bm,5))+","+str(round(mx,3)))
    w.close()    

def obtain_threshold2(p,m,ts,x):
    a = []
    for i in range(len(ts)):
        dt=ts[i].split(" ")
        a.append(ind.indicator(p,"lookahead",tf,0,dt[0],dt[1],0))

    lots = m[p].predict(x)

    #c1 = p[:3]
    #c2 = p[-3:]
    #if c1=="USD":
    #    lots = m[c2].predict(x)
    #    for i in range(len(lots)): lots[i] = lots[i]
    #elif c2=="USD":
    #    lots = m[c1].predict(x)
    #else:
    #    cl1 = m[c1].predict(x)
    #    cl2 = m[c2].predict(x)
    #    lots=[]
    #    if (c1+"USD" in pairs99)       and     (c2+"USD" in pairs99): 
    #        for i in range(len(cl1)): lots.append(cl1[i]-cl2[i])
    #    if (not (c1+"USD" in pairs99)) and     (c2+"USD" in pairs99): 
    #        for i in range(len(cl1)): lots.append(-cl1[i]-cl2[i])
    #    if (c1+"USD" in pairs99)       and (not (c2+"USD" in pairs99)): 
    #        for i in range(len(cl1)): lots.append(cl1[i]+cl2[i])
    #    if (not (c1+"USD" in pairs99)) and (not (c2+"USD" in pairs99)): 
    #        for i in range(len(cl1)): lots.append(-cl1[i]+cl2[i])

    #dt=ts[0].split(" ")
    #print(p+":"+str(ind.indicator(p,"lookahead",tf,0,dt[0],dt[1],7))+":"+str(sum(a[:8])))
    mx = -999999999
    bc = 0
    bcc = 0
    sc = 0
    scc = 0
    bB = None
    bm = 0
    debug = False
    factor = 1000000
    for j in range(500):
        t = 0.1 + float(j)/100.0
        pf,m,wRate,cnt,pp,rpp,B = simulate4(lots,a,p,t)
        if (debug): print(p+":"+str(t)+":   "+str(round(pf,4))+":"+str(cnt)+":"+str(round(m,3)))
        if cnt<0:break
        if pf>mx:
            mx = pf
            bt = t
            bB = B
            bm = m
    #for j in range(1,10000):
    #    t = j / factor
    #    balance,m,wRate,cnt,pp,rpp,B = simulate3(lots,a,p,t,0,-t,0)
    #    if (debug): print(p+":"+str(t)+":   "+str(round(balance*m,2))+":"+str(cnt)+":"+str(round(m,3)))
    #    if cnt<0:break
    #    if balance*m>mx:
    #        mx = balance*m
    #        bc = t
    #        sc = -t
    #        bB = B
    #        bm = m
    #for j in range(1,3000):
    #    t = 0 - j / factor
    #    balance,m,wRate,cnt,pp,rpp,B = simulate3(lots,a,p,bc,t,sc,-t)
    #    if (debug): print(p+":"+str(t)+":   "+str(round(balance*m,2))+":"+str(cnt)+":"+str(round(m,3)))
    #    if balance*m>mx:
    #        mx = balance*m
    #        bcc= t
    #        scc=-t
    #        bB = B
    #        bm = m
    #mx = -999999999            
    #for j in range(1,10000):
    #    t = 0 - j / factor
    #    balance,m,wRate,cnt,pp,rpp,B = simulate3(lots,a,p,9,0,t,0)
    #    if (debug): print(p+":"+str(t)+":   "+str(round(balance*m,2))+":"+str(cnt)+":"+str(round(m,3)))
    #    if cnt<20:break
    #    if balance*m>mx:
    #        mx = balance*m
    #        sc = t
    #        bB = B
    #        bm = m
    #for j in range(1,3000):
    #    t = j / factor
    #    balance,m,wRate,cnt,pp,rpp,B = simulate3(lots,a,p,bc,bcc,sc,t)
    #    if (debug): print(p+":"+str(t)+":   "+str(round(balance*m,2))+":"+str(cnt)+":"+str(round(m,3)))
    #    if balance*m>mx:
    #        mx = balance*m
    #        scc= t
    #        bB = B
    #        bm = m
    #tc = max(bc,-sc)
    #sc = min(-bc,sc)
    #bc = tc
    #balance,bm,wRate,cnt,pp,rpp,B = simulate3(lots,a,p,bc,bcc,sc,scc)
    plt.plot(np.array(range(len(bB))), np.array(bB), label= '3x')
    plt.savefig("combi_"+str(tf)+"/"+p+"_testing_balance.png")
    plt.close()                                            
    #print(p+" best tresholds: "+str(round(mx))+"  :  \t"+str(bc)+"\t"+str(bcc)+"\t"+str(sc)+"\t"+str(scc))
    print(p+" best tresholds: "+str(round(mx,5))+"  :  \t"+str(bt))
    w = open("combi_"+str(tf)+"/"+p+".index",'w')
    w.write(str(bt)+","+str(round(bm,5))+","+str(round(mx,3)))
    w.close()    

def reset_sum():
    w = open("combi_"+str(tf)+"/sum.txt",'w')
    w.write("0")
    w.close()

def test(p,m,ts,x,index,t):
    def add_sum(num):
        r = open("combi_"+str(tf)+"/sum.txt",'r')
        d = list(csv.reader(r, quoting=csv.QUOTE_NONNUMERIC))
        r.close()
        w = open("combi_"+str(tf)+"/sum.txt",'w')
        w.write(str(float(d[0][0])+num))
        w.close()
        print("SUM added: "+str(round(d[0][0]+num))+"("+str(round(num))+")")

    a = []
    for i in range(len(ts)):
        dt=ts[i].split(" ")
        a.append(ind.indicator(p,"lookahead",tf,0,dt[0],dt[1],0))
    c1 = p[:3]
    c2 = p[-3:]
    if c1=="USD":
        lots = m[c2].predict(x)
        for i in range(len(lots)): lots[i] = -lots[i]
    elif c2=="USD":
        lots = m[c1].predict(x)
    else:
        cl1 = m[c1].predict(x)
        cl2 = m[c2].predict(x)
        lots=[]
        for i in range(len(cl1)): lots.append(cl1[i]-cl2[i])

    r = open("combi_"+str(tf)+"/"+p+'.index','r')
    d = list(csv.reader(r, quoting=csv.QUOTE_NONNUMERIC))
    r.close()
    #print(p+":"+str(bc))
    #balance,m,wRate,cnt,pp,rpp,B = simulate(lot,a,p)
    #balance,m,wRate,cnt,pp,rpp,B = simulate3(lots,a,p,d[0][0]              ,d[0][1],d[0][2]              ,d[0][3])
    balance,m,wRate,cnt,pp,rpp,B = simulate3(lots,a,p,max(d[0][0],-d[0][2]),d[0][1],min(-d[0][0],d[0][2]),d[0][3])
    #balance,m,wRate,cnt,pp,rpp,B = simulate3(lots,a,p,max(d[0][0],-d[0][2]), max(d[0][1],-d[0][3]),   min(-d[0][0],d[0][2]),   min(-d[0][1],d[0][3]))
    #balance,m,wRate,cnt,pp,rpp,B = simulate3(lots,a,p,(d[0][0]-d[0][2])/2  , (d[0][1]-d[0][3])/2  ,   (-d[0][0]+d[0][2])/2 ,   (-d[0][1]+d[0][3])/2)
    #balance,m,wRate,cnt,pp,rpp,B = simulate3(lots,a,p,d[0][0]*1.5          ,d[0][1],d[0][2]*1.5          ,d[0][3])
    #balance,m,wRate,cnt,pp,rpp,B = simulate3(lots,a,p,0,0,0,0)
    fig = plt.figure()
    plt.plot(np.array(range(len(B))), np.array(B), label= '3x')
    plt.savefig("combi_"+str(tf)+"/"+p+"_final_testing_balance.png")
    plt.close()    
    #m=d[0][1]
    #print(p+": TEST : "+str(round(balance))+"  "+str(round(m,3))+ " lots")
    w = open("combi_"+str(tf)+"/"+p+".lots",'w')
    w.write(str(m))
    w.close()    
    add_sum(balance*m)
    return balance , B, m, cnt

def get_models():
    models = [{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}]
    bms = [{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}]
    bc = [{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}]
    bcc = [{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}]
    sc = [{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}]
    scc = [{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}]
    score = [{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}]

    for p in pairs99:
        #if "JPY" in p: continue
        for s in range(16):
            acc = []
            files=glob(d+"/"+p+"."+str(s)+"_*regressor.accuracy")
            mx = -999999999
            mf = ""
            ms = -1
            for f in files:
                r = open(f,'r')
                ln = r.readline()
                acc = float(ln.split(",")[0])
                m = float(ln.split(",")[1])
                r.close()
                if acc>mx:
                    if tf<100: ms=f[34:36]
                    else: ms=f[35:37]
                    if ms[-1]=="_": ms = int(ms[:-1])
                    else          : ms = int(ms)
                    if ms>18: continue
                    mx=acc
                    mf=f
                    bm=m
            if mx>-999999:
                print(mf+" chosen: "+str(ms)+" : "+str(mx)+" : "+str(round(bm,3)))
                r = open(mf[:-9],'rb')
                models[s][p]=pickle.load(r)
                r.close()
                bms[s][p] = bm
                score[s][p] = mx
                #csvfile = open(mf[:-9]+".index", 'r',newline='')
                #ind = list(csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC))[0]
                #csvfile.close()
                #bc[s][p]    = ind[0]
                #bcc[s][p]   = ind[1]
                #sc[s][p]    = ind[2]
                #scc[s][p]   = ind[3]
                #score[s][p] = ind[5]
    return models,bms,score

def get_all_models():
    models = {}
    for p in pairs99:
        #if "JPY" in p: continue
        files=glob(d+"/"+p+"*regressor.accuracy")
        models[p] = []
        for mf in files:
            print(mf)
            r = open(mf[:-9],'rb')
            models[p].append(pickle.load(r))
            r.close()
    return models

def get_l1_models():
    models = []
    for c in ["JPY","EUR","GBP","AUD","CAD","CHF","NZD"]:
        acc = []
        for s in range(12):
            files=glob(d+"/"+c+"."+str(s)+"_*accuracy")
            mx = -999999
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
            if mx>-99999:
                print(mf+" chosen: "+str(ms)+" : "+str(mx))
                r = open(mf[:-9],'rb')
                models.append(pickle.load(r))
                r.close()
    return models

def get_l2_models():
    models = {}
    for c in ["JPY","EUR","GBP","AUD","CAD","CHF","NZD"]:
        files=glob(d+"/"+c+".*2.accuracy")
        mx = -999
        mf = ""
        for f in files:
            r = open(f,'r')
            acc = float(r.readline())
            r.close()
            if acc>mx:
                mx=acc
                mf=f
        if mx>-99:
            print(mf+" chosen: "+str(mx))
            r = open(mf[:-9],'rb')
            models[c]=pickle.load(r)
            r.close()
    return models

def get_very_best_models():
    models = {}
    for c in curr2:
        mx = -999
        mf = ""
        ms = -1
        me = -999999
        mt = -1
        mm = -1
        for t in [120,240,360,480]:
            xf,ts,x,y = load_combi("",t)
            ts=ts[-int(60/t*24*23*mtt):]
            x = x[-int(60/t*24*23*mtt):]
            dd = "combi_"+str(t)+"/gradient_boosting"
            files=glob(dd+"/"+c+"*regressor.accuracy")
            for f in files:
                r = open(f,'r')
                acc = float(r.readline())
                r.close()
                r = open(f[:-9],'rb')
                mo = pickle.load(r)
                r.close()
                earning,B,m,cnt = test(c,mo,ts,x,0,t)
                print(f+": "+str(ms)+" : "+str(acc) + " : "+str(round(earning,1))+"("+str(cnt)+")")
                if earning>me and earning<10000000:
                    if tf<100: ms=f[31:33]
                    else: ms=f[32:34]
                    if ms[-1]=="_": ms = int(ms[:-1])
                    else          : ms = int(ms)
                    if ms>18: continue
                    me=earning
                    mx=acc
                    mf=f
                    mt=t
                    mm=m
        if me>-99:
            r = open(mf[:-9],'rb')
            models[c]=pickle.load(r)
            r.close()
            t = int(mf[6:9])
            xf,ts,x,y = load_combi("",t)
            ts=ts[-int(60/mt*24*23*mtt):]
            x = x[-int(60/mt*24*23*mtt):]
            earning,B,m,cnt = test(c,models[c],ts,x,0,mt)
            fig = plt.figure()
            plt.plot(np.array(range(len(B))), np.array(B), label= '3x')
            plt.ticklabel_format(style='plain')
            plt.ticklabel_format(useOffset=False)
            plt.savefig("final_selection/"+c+".png")
            plt.close()
            w = open("final_selection/"+c+".lots",'w')
            w.write(str(m)+","+str(mt))
            w.close()                
            print(mf+" chosen: "+str(ms)+" : "+str(mx) + " : "+str(round(me))+"("+str(cnt)+")\n\n\n\n")
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
    #print(x1[:5])
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

def save_validation():
    with gzip.open("data/combi"+str(tf)+".x2", 'wb') as f:
        ubjson.dump([xf,ts,x2,y[0]], f)

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
        yy = [{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}]
        yt = [{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}]

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
        xx = scaler.fit_transform(xx)
        xt = scaler.fit_transform(xt)
        for shift in range(16):
            for p in pairs99:
                if "JPY" in p: continue
                yy[shift][p] = y[shift][p][:-validation]
                yt[shift][p] = y[shift][p][-validation:]
                print(p+" "+str(shift)+": Train Test Split X_33333 Veritication: " + str(xx[3333][0])+"   Y:"+str(yy[shift][p][3333]))
        pool = mp.Pool(1)
        #for f in glob(d+"/*tmp"):
        #    r = open(f,'rb')
        #    d = pickle.load(r)
        #    r.close()
        #    if   f[32]=="_": pl.append(pool.apply_async(resume_gb_train, (int(f[31:32]),f[27:30],f[27:-4],d[0],d[1],d[2],d[3],d[4])))
        #    else           : pl.append(pool.apply_async(resume_gb_train, (int(f[31:33]),f[27:30],f[27:-4],d[0],d[1],d[2],d[3],d[4])))
        for iter in range(iterations):
            for shift in range(16):
                #for p in ["GBPUSD","AUDUSD","GBPCAD","GBPAUD","NZDCAD","AUDNZD"]:#pairs99:
                #for p in ["GBPAUD","GBPUSD","AUDNZD"]:#pairs99:
                for p in pairs99:
                    if "JPY" in p: continue
                    #pl.append(pool.apply_async(gb_train, (shift,p)))
                    pl.append(pool.apply_async(rf_train, (shift,p)))
                    #nn_train(shift,p)
                    #xg_train(shift,p)
        for pl2 in pl: pl2.get()
    elif argv[1]=="simulate":
        simulate()
    elif argv[1]=="prepare_l2":
        xf = {}
        models = get_l1_models()        
        print(str(len(models))+" models loaded")
        xf,ts,x,y = load_combi()
        #while (len(x[0])!=5348): 
        #    ts = ts[1:]
        #    x = x[1:]
        #    y = y[1:]
        #   print("skipping...")
        print("combi loaded. predicting...")
        x2=[]
        for xx in x:
            xx2=[]
            for m in models: 
                xx2.append(m.predict([xx])[0])
            x2.append(xx2)
            if len(x2)%100==0: print(len(x2))
        save_validation()
        print("validation saved")
        x=[]
    elif argv[1]=="train_l2":
        xf,ts,x,yy= load_combi()
        xf,ts,x,y = load_combi("2")
        a = {}
        for c in curr2:
            a[c] = []
            if (c+"USD") in pairs99: p = c+"USD"
            else:                    p = "USD"+c
            for i in range(len(ts[-validation:])):
                dt=ts[-validation:][i].split(" ")
                a[c].append(ind.indicator(p,"lookahead",tf,0,dt[0],dt[1],0))        
        yy = {}
        yt = {}
        for c in curr2:
            xx = x[:-validation]
            xt = x[-validation:-testing]
            yy[c] = y[c][:-validation]
            yt[c] = y[c][-validation:-testing]
        pool = mp.Pool(20)
        for c in curr2: pl.append(pool.apply_async(gb_train_l2, (c,)))
        for pl2 in pl: pl2.get()
    elif argv[1]=="obtain_threshold":
        if True:
        #if False:
            models = get_models()
            xf,ts,x,y = load_combi()
        else:
            models = get_l2_models()
            xf,ts,x,y = load_combi("2")
        ts=ts[-validation:]
        x = x[-validation:]
        y = []
        #pool = mp.Pool(28)
        #for p in ["GBPAUD","GBPUSD","AUDNZD"]:#pairs99:
        for p in pairs99:
            if "JPY" in p: continue
            #pl.append(pool.apply_async(obtain_threshold3,(p,models,ts,x)))
            obtain_threshold3(p,models,ts,x)
        for pl2 in pl: pl2.get()    
    elif argv[1]=="test":
        if True:
        #if False:
            models = get_models()
            xf,ts,x,y = load_combi()
        else:
            models = get_l2_models()
            xf,ts,x,y = load_combi("2")
        ts=ts[-testing:]
        x = x[-testing:]
        y = []
        pool = mp.Pool(28)
        for p in pairs99:
            if "JPY" in p : continue
            pl.append(pool.apply_async(test,(p,models,ts,x,1,tf)))
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