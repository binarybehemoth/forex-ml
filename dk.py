checkback_years = 20

import lzma
import struct
import pandas as pd
from sys import argv
from pathlib import Path
import queue, threading
import multiprocessing as mp
from symbols import *
import urllib.request
import indicators as ind

def bi5_to_df(filename, fmt):
    chunk_size = struct.calcsize(fmt)
    data = []
    with lzma.open(filename) as f:
        while True:
            try:
                chunk = f.read(chunk_size)
            except:
                break
            if chunk:
                data.append(struct.unpack(fmt, chunk))
            else:
                break
    df = pd.DataFrame(data)
    return df

def get_day(p,dt):
    msec = []
    bid = []
    for i in range(24):
        f="Data/"+p+"/"+str(int(dt/10000))+"/"+str(int((dt%10000)/100)).zfill(2)+"/"+str(dt%100).zfill(2)+"/"+str(i).zfill(2)+"h_ticks.bi5"
        #print(f)
        if Path(f).is_file() and Path(f).stat().st_size>0:
            data = bi5_to_df(f, '>3i2f')
            try:
                for j in range(len(data[0])): msec.append(data[0][j]+i*3600*1000)
                for j in range(len(data[2])): bid.append(data[2][j]/unit[p])
            except: 
                print(f+" error")
                continue
        else:
            continue
    return msec,bid

def parse_write(w,dt,hh,nn,ss,msec,bid,tf,ohr,oo,oh,ol,oc):
    i=0
    #start=msec[0]-(msec[0]%(tf*60000))
    start=hh*3600000 + nn*60000
    if start>msec[-1]:return ohr,oo,oh,ol,oc
    while msec[i]<start: i=i+1
    nxt = start+tf*60000
    yr = int(dt/10000)
    mo = int((dt%10000)/100)
    dy = dt%100    
    while True:
        hr = int(start/3600/1000)
        mn = int((start%3600000)/60000)
        #print(str(i)+":"+str(yr)+":"+str(mo)+":"+str(dy)+":"+str(hr)+":"+str(mn))
        o=bid[i]
        h=o
        l=o
        while msec[i]>=start and msec[i]<nxt:
            if bid[i]>h: h=bid[i]
            if bid[i]<l: l=bid[i]
            i+=1
            if i==len(bid):break
        c=bid[i-1]
        if (hr>0 or (hr==0 and ohr!=20)) and (oo!=o or oh!=h or ol!=l or oc!=c): 
            #print(str(yr*10000+mo*100+dy+100)+","+str(hr)+":"+str(mn)+":00,"+str(o)+","+str(h)+","+str(l)+","+str(c)+",0\n")
            w.write(str(yr*10000+mo*100+dy+100)+","+str(hr).zfill(2)+":"+str(mn).zfill(2)+":00,"+str(o)+","+str(h)+","+str(l)+","+str(c)+",0\n")
        ohr=hr
        oo=o
        oh=h
        ol=l
        oc=c
        if i==len(bid):return ohr,oo,oh,ol,oc
        start=nxt
        nxt=start+tf*60000

def update(p,tf):
    ny,nm,nd,nh,nn,ns = today_datetime()
    end = ny*10000 + nm*100 + nd + 1
    try:
        y,m,d,h,n,s = get_last_csv_datetime(p,tf)
    except Exception as e:
        print(p+": ERROR ------------------------------------------------------- "+str(e))
        return
    ohr=h
    n = n+tf
    if n>=60:
        h = h+int(n/60)
        n = n%60
    if h>=24:
        d = d+int(h/24)
        h = h%24
    while not is_valid_date(y,m,d):
        d = d+1
        if d>31:
            d=1
            m=m+1
        if m>=12:
            y=y+1
            m=0
    w = open("data/"+p+str(tf)+".csv",'a')
    oo=0
    oh=0
    ol=0
    oc=0
    while True:
        dt = y*10000 + m*100 + d
        #if (dt%100==1): 
        if dt>end: break
        print(p+": processing "+str(dt+100)+"......"+str(tf))
        msec, bid = get_day(p,dt)
        if len(msec)>1 and msec[0]>-1: 
            ohr,oo,oh,ol,oc=parse_write(w,dt,h,n,s,msec,bid,tf,ohr,oo,oh,ol,oc)
        h=0
        n=0
        s=0
        d=d+1
        if d==32: 
            d=1
            m=m+1
        if m==12: 
            m=0
            y=y+1
    w.close()

def get_last_csv_datetime(p,tf):
    if not Path('data/'+str(p)+str(tf)+'.csv').is_file():
        if p in ["USATECHIDXUSD"]: return 2013,1,1,0,0,0
        elif p in exotics: return 2013,1,1,0,0,0
        else:            return 2007,1,1,0,0,0
    r = open('data/'+str(p)+str(tf)+'.csv','r')
    while True:
        d,t,o,h,l,c,a,v = parseLine(r)
        #print(d)
        if (d==-1): break
        dd=d
        tt=t
    r.close()
    return int(dd/10000),int((dd%10000)/100)-1,dd%100,int(tt[:2]),int(tt[3:5]),int(tt[6:])

def is_valid_date(year, month, day):
    day_count_for_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if year%4==0 and (year%100 != 0 or year%400==0):
        day_count_for_month[1] = 29
    return (0 <= month <= 11 and 1 <= day <= day_count_for_month[month])

def download(p):
    yy,m,dd,h,n,s = today_datetime()
    y = yy
    d = dd
    while y > 2016:
        while not is_valid_date(y,m,d):
            d = d-1
            if d<=0:
                d=31
                m=m-1
            if m==-1:
                m=11
                y=y-1
        di  = p+"/"+str(y)+"/"+str(m).zfill(2)+"/"+str(d).zfill(2)+"/"
        do = 'data/'+di               
        if not os.path.exists(do): os.makedirs(do)
        for h in range(24):
            f = str(h).zfill(2)+'h_ticks.bi5'
            try :
                if not Path(do+f).is_file(): 
                    urllib.request.urlretrieve('http://www.dukascopy.com/datafeed/'+di+f, do+f)
                    print(do+f+" downloaded")
            except:
                if d==dd: 
                    #print('ending at '+do+f)
                    break
                print("retrying "+do+f+".......")
                try :
                    if not Path(do+f).is_file(): 
                        urllib.request.urlretrieve('http://www.dukascopy.com/datafeed/'+di+f, do+f)
                        print(do+f+" downloaded")
                except:
                    print('skipping at '+do+f+'.....................................................................ERROR!')
                    break    
        if yy-checkback_years==y: 
            print(p+" done!")
            break
        d = d-1

if __name__ == '__main__':
    targets = ["USA500IDXUSD"]
    print(str(len(stocks))+" stocks")
    pairs = stocks
    if argv[1]=="download":
        pool = mp.Pool(8)
        pl=[]
        for i in range(5):      # repeated to protect against occassional failed downloads
            #for p in stocks: 
            for p in targets:
            #for p in pairs99:# + chosen:
                #if p=="US500": p = "USA500IDXUSD"
                #else: p = p+"USUSD"
                pl.append(pool.apply_async(download,[p]))
        for pl2 in pl: pl2.get()
    elif argv[1]=="update":
        pool = mp.Pool(4)
        pl=[]
        for p in targets:
        #for p in pairs99:# + chosen:
            pl.append(pool.apply_async(update,(p,int(argv[2]))))
            #pl.append(pool.apply_async(update,(p,1)))
            #pl.append(pool.apply_async(update,(p,5)))
            #pl.append(pool.apply_async(update,(p,15)))
            #pl.append(pool.apply_async(update,(p,30)))
            #pl.append(pool.apply_async(update,(p,60)))
            #pl.append(pool.apply_async(update,(p,240)))
        for pl2 in pl: pl2.get()