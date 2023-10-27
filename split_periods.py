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
    y=[]
    for i in range(int(24*60/tf)): x.append([])
    for i in range(int(24*60/tf)): y.append([])
    yo=[]
    cnt=0
    for i in range(50):
        d,t,o,h,l,c,a,v = ind.parseLine(r,divisor)
        oo = o
        w.append(o)
    while True:
        xx=[]
        xx.append(c-l)
        xx.append(h-c)
        d,t,o,h,l,c,a,v = ind.parseLine(r,divisor)
        if d==-1: break
        for i in [1,2,3,4,5,6,7,8,10,12,14,16,20,25,30,40,50]:
            xx.append(o-w[50-i])
        w = w[1:]
        w.append(o)
        prd = int(float(t.split(":")[0])*4)+int(int(t.split(":")[1])/tf)
        x[prd].append(xx)
        la = ind.indicator(p,"lookahead",tf,0,d,t,0)
        th = 0 #commission[p]/unit[p]+spread[p]
        if la > o*0.001: la = 3
        elif la > o*0.0005: la = 2
        elif la > 0: la=1
        elif la < -o*0.001: la = -3
        elif la < -o*0.0005: la = -2
        elif la < 0: la=-1
        else: la=0
        y[prd].append(la)
        cnt+=1
    r.close()
    ind.data={}
    print(p+": "+str(len(x))+" "+str(len(x[0]))+" "+str(len(x[0][0])))
    with gzip.open("data/"+p+str(tf)+".s0", 'wb') as f: pickle.dump([x,y], f)          
    print(p+" closed with "+str(cnt)+ " records...")
    print(len(x[0][0]))
    print(len(x[0]))
    print(p+" done!")

cache = {}

def pload2(file):
    print("loading..."+file)
    r = gzip.open(file,'rb')
    d = pickle.load(r)
    r.close()
    print(file+" loaded ")
    return d[0],d[1]
    
def gb_train(p):
    lr = 0.1
    best_score=-9999999
    best_regressor=None
    LV = uniform(0.002,0.01)
    SP = uniform(0.01,0.05)
    SS = 1
    r=GradientBoostingRegressor(verbose=0, warm_start=True, loss='huber', learning_rate=lr, n_estimators=1,max_depth=3,min_samples_leaf=LV,min_samples_split=SP,subsample=SS, alpha=0.9)
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
        score,trades, mx, mn = simulate(r.predict(xt[p]),yt[p],p)
        if score>best_score: 
            best_score=score
            best_score_trades = trades
            best_regressor=copy.deepcopy(r)
            bj=j
        if j-bj>9: break
    if not os.path.exists(d): os.makedirs(d)
    best_score,best_score_trades, mx, mn = simulate(best_regressor.predict(xtt[p]),ytt[p],p)
    return best_score, best_score_trades, best_regressor, mx, mn


def simulate(pred,actual,pair):
    cnt = len(actual)
    th = (max(pred)-min(pred))/2*0.1
    trades = 0
    correct = 0
    mx = max(pred)
    mn = min(pred)
    for i in range(4,len(actual)):
          trades=trades+1
          if pred[i]*actual[i]>0: correct = correct + 1
    return correct/trades, trades, mx, mn

if __name__ == '__main__':
    #pairs99 = ["EURUSD","GBPUSD","USDCHF","USDCAD","AUDUSD","NZDUSD"]
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
    ytt={}
    iter = 20
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
            print(p)
            pl.append(pool.apply_async(prepare,(p,tf)))
        for pl2 in pl: pl2.get()
    elif argv[1]=="combi":
        save_combi()
    elif argv[1]=="train":
        tscore = 0
        def evaluate_accuracy(p):
            x[p],y[p] = pload2("Data/"+p+str(tf)+".s0")
            #if "JPY" in p: continue
            mm=[]
            ww=[]
            score=[]
            for i in range(int(24*60/tf)): score.append(0)
            sw=0
            imp = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            for ii in range(iter):
                for q in range(13):
                #for q in range(12,13):
                    for period in range(int(24*60/tf)):
                        mon = 23
                        last = mon*((14-q)*3)+1
                        xx[p] = x[p][period][q*3*mon:-last]
                        xt[p] = x[p][period][-last:-last+mon*3]
                        xtt[p]= x[p][period][-last+mon*3:-last+mon*6]
                        yy[p] = y[p][period][q*3*mon:-last]
                        yt[p] = y[p][period][-last:-last+mon*3]
                        ytt[p]= y[p][period][-last+mon*3:-last+mon*6]
                        s, t, r, m, n = gb_train(p)
                        score[period] = score[period] + s
                    for period in range(int(24*60/tf)):
                       ave = score[period]/(ii*13+q+1)
                       if ave>0.55 and period>4 and period<95:print(">>>........................................................................................................"+p+"..."+str(ii)+"..."+str(q)+"..."+str(period)+" : "+str(ave))
            with gzip.open("data/"+p+str(tf)+".score", 'wb') as f: pickle.dump(score, f)
        pr = []
        for p in pairs99:
            if "JPY" in p : continue
            pr.append(mp.Process(target=evaluate_accuracy, args=(p,)))
            pr[-1].start()
        for j in range(len(pr)): pr[j].join()
    elif argv[1]=="generate":
        files = glob("data/selected/*.regressor")
        for f in files: os.remove(f)
        def obtain_best(p):
            with gzip.open("data/"+p+str(tf)+".score", 'rb') as f: score=pickle.load(f)   
            x[p],y[p] = pload2("Data/"+p+str(tf)+".s0")
            for period in range(int(24*60/tf)):
               if score[period]/iter/13<0.55 or period<4 or period>94: continue
               mon = 23
               xx[p] = x[p][period][3*mon:-3*mon]
               xt[p] = x[p][period][-3*mon:]
               xtt[p]= x[p][period][-3*mon:]
               yy[p] = y[p][period][3*mon:-3*mon]
               yt[p] = y[p][period][-3*mon:]
               ytt[p]= y[p][period][-3*mon:]
               bs = 0
               bt = 0
               bm = 0
               for i in range(10):
                    s, t, r, m, n = gb_train(p)
                    if s>bs:
                        bs = s
                        bt = t
                        bm = m
                        bn = n
                        br = r
               if bs<0.55: continue                        
               print(p+":"+str(period)+"---" +str(bs)+"\t"+str(bt))
               with gzip.open("data/selected/"+p+str(tf)+"_"+str(period)+"_"+str(bm)+"_"+str(bn)+".regressor", 'wb') as f: 
                   pickle.dump(br, f)
                   print("data/selected/"+p+str(tf)+"_"+str(period)+"_"+str(bm)+"_"+str(bn)+".regressor...saved!")
        pr = []
        for p in pairs99:
            if "JPY" in p : continue
            #if p == "AUDCAD": continue
            pr.append(mp.Process(target=obtain_best, args=(p,)))
            pr[-1].start()         
    elif argv[1]=="export":
        def wr(w,d): w.write(str(d)+"\r\n")
        def wrtree(w,t):
            wr(w,len(t.feature))
            #print(len(t.feature))
            for i in range(len(t.feature)):
               wr(w,t.children_left[i])
               wr(w,t.children_right[i])
               wr(w,t.feature[i])
               wr(w,t.threshold[i])
               wr(w,t.value[i][0][0])
        files = glob("data/selected/*.regressor")
        w = open('/mnt/c/Users/Lip Phang/AppData/Roaming/MetaQuotes/Terminal/Common/Files/all.regressors', 'w',newline='')
        wr(w,len(files))
        for f in files:
           print(f)
           with gzip.open(f, 'rb') as fr: model = pickle.load(fr)
           wr(w,f.split("/")[2][:6])
           wr(w,f.split("_")[1])
           wr(w,f.split("_")[2].split(".")[0]+"."+f.split("_")[2].split(".")[1])
           wr(w,f.split("_")[3].split(".")[0]+"."+f.split("_")[3].split(".")[1])
           print(f.split("_")[3].split(".")[0]+"."+f.split("_")[3].split(".")[1])
           wr(w,model.learning_rate)
           wr(w,model.init_.predict([0])[0])
           wr(w,len(model.estimators_))
           for tree in model.estimators_: wrtree(w,tree[0].tree_)
        wr(w,"END")
        w.close()
