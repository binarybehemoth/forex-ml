from symbols import *
from copy import deepcopy

data = {}
#test = "_test"
test = ""
i_start = 19270701

def indicator(pair,name,timeframe,period,date,time,shift=0):
    k1 = pair+'_'+name+'_'+str(timeframe)+'_'+str(period)
    if k1 not in data:
        if name=='ema':
            data[k1] = prepare_ema(pair,timeframe,period)
        if name=='ema2':
            data[k1] = prepare_ema2(pair,timeframe,period)
        if name=='ema3':
            data[k1] = prepare_ema3(pair,timeframe,period)
        elif name=='wpr':
            data[k1] = prepare_wpr(pair,timeframe,period)
        elif name=='sd':
            data[k1] = prepare_sd(pair,timeframe,period)
        elif name=='bb':
            data[k1] = prepare_bb(pair,timeframe,period)
        elif name=='psar':
            data[k1] = prepare_psar(pair,timeframe,period)
        elif name=='breakout':
            data[k1] = prepare_breakout(pair,timeframe,period)
        elif name=='cci':
            data[k1] = prepare_cci(pair,timeframe,period)
        elif name=='rsi':
            data[k1] = prepare_rsi(pair,timeframe,period)
        elif name=='atr':
            data[k1] = prepare_atr(pair,timeframe,period)
        elif name=='macd':
            data[k1] = prepare_macd(pair,timeframe,period)
        elif name=='macd_signal':
            data[k1] = prepare_macd_signal(pair,timeframe,period)
        elif name=='demarker':
            data[k1] = prepare_demarker(pair,timeframe,period)
        elif name=='force':
            data[k1] = prepare_force(pair,timeframe,period)
        elif name=='trix':
            data[k1] = prepare_trix(pair,timeframe,period)            
        elif name=='adx':
            data[k1] = prepare_adx(pair,timeframe,period)
        elif name=='momentum':
            data[k1] = prepare_momentum(pair,timeframe,period)
        elif name=='rvi':
            data[k1] = prepare_rvi(pair,timeframe,period) 
        elif name=='strength':
            data[k1] = prepare_strength(pair,timeframe,period)                        
        if name=='lookahead':
            data[k1] = prepare_lookahead(pair,timeframe)
        if name=='ideal_lot':
            data[k1] = prepare_ideal_lot(pair,timeframe)            
        if name=='abcd':
            data[k1] = prepare_abcd(pair,timeframe,period)
        if name=='firstHit':
            data[k1] = first_hit(pair,timeframe,period)
        print(k1+' prepared')
    k2 = str(date)+' '+time
    if k2 not in data[k1]:
        if k2 == str(date)+' '+'00:00:00':
            k2 = str(date)+' '+'01:00:00'
    if k2 not in data[k1]:
        if k2 == str(date)+' '+'01:00:00':
            k2 = str(date)+' '+'02:00:00'
    if k2 not in data[k1]:
        if name=="adx" or name=="rvi":
            return [-999,-999]
        elif name=="abcd":
            return [-999,-999,-999,-999]
        else:
            #print(pair+" : no data found on "+k1+"....."+k2)
            return -999
    if name=='lookahead':
        d = data[k1][k2]
        o = d[0][0]
        for i in range(shift):
            d = d[1]
            if (len(d)<2): return 0
        c = d[0][1]
        return c-o
    else:
        d = data[k1][k2]
        try:
            for i in range(0,shift): d = d[1]
        except:
            if name=="adx" or name=="rvi":
                return [-999,-999]
            elif name=="abcd":
                return [-999,-999,-999,-999]
            else:
                return -999
        if len(d)<1: 
            if name=="adx" or name=="rvi":
                return [-999,-999]
            elif name=="abcd":
                return [-999,-999,-999,-999]
            else:
                return -999
        return d[0]

def parseLine(r,d=1):
    l = r.readline()
    if not l: return -1,-1,-1,-1,-1,-1,-1,-1
    s = l.split(',')
    while (int(s[0])<i_start):
        l = r.readline()
        if not l: return -1,-1,-1,-1,-1,-1,-1,-1
        s = l.split(',')
    o = float(s[2])/d
    h = float(s[3])/d
    l = float(s[4])/d
    c = float(s[5])/d
    e = s[6].find('\n')==-1
    return int(s[0]),s[1],o,h,l,c,(h+l+c*2)/4,float(s[6])

def prepare_ema(p,t,period):
    r = open('data/'+str(p)+str(t)+test+'.degap.csv','r')
    r.readline()
    sum=0
    for i in range(0,period):
        d,t,o,h,l,c,a,v = parseLine(r)
        sum+=a
    last = sum/period
    m = 2 / (period + 1) 
    result={}
    prev=[]
    while True:
        d,t,o,h,l,c,a,v = parseLine(r)
        if (d==-1): break
        last  = (a-last)*m+last
        result[str(d)+' '+t] = [last,prev]
        prev = [last,prev]
    r.close()
    return result

def prepare_ema2(p,tf,period):
    r = open('data/'+str(p)+str(tf)+test+'.degap.csv','r')
    r.readline()
    for i in range(0,period): d,t,o,h,l,c,a,v = parseLine(r)
    last = indicator(p,"ema",tf,period,d,t,0)
    m = 2 / (period + 1) 
    result={}
    prev=[]
    while True:
        d,t,o,h,l,c,a,v = parseLine(r)
        if (d==-1): break
        e = indicator(p,"ema",tf,period,d,t,0)
        last  = (e-last)*m+last
        result[str(d)+' '+t] = [last,prev]
        prev = [last,prev]
    r.close()
    return result

def prepare_ema3(p,tf,period):
    r = open('data/'+str(p)+str(tf)+test+'.degap.csv','r')
    r.readline()
    for i in range(0,period): d,t,o,h,l,c,a,v = parseLine(r)
    last = indicator(p,"ema2",tf,period,d,t,0)
    m = 2 / (period + 1) 
    result={}
    prev=[]
    while True:
        d,t,o,h,l,c,a,v = parseLine(r)
        if (d==-1): break
        e = indicator(p,"ema2",tf,period,d,t,0)
        last  = (e-last)*m+last
        result[str(d)+' '+t] = [last,prev]
        prev = [last,prev]
    r.close()
    return result

def prepare_wpr(p,t,period):
    r = open('data/'+str(p)+str(t)+test+'.degap.csv','r')
    r.readline()
    highs=[]
    lows=[]
    for i in range(0,period):
        d,t,o,h,l,c,a,v = parseLine(r)
        highs.append(h)
        lows.append(l)
    result={}
    prev=[]
    while True:
        d,t,o,h,l,c,a,v = parseLine(r)
        if (d==-1): break
        highs = highs[1:]
        highs.append(h)
        lows = lows[1:]
        lows.append(l)
        mx = max(highs)
        mn = min(lows)
        if mx==mn:
            value = -50
        else:
            value = (mx-c)/(mx-mn)*(-100)
        result[str(d)+' '+t] = [value,prev]
        prev = [value,prev]
    r.close()
    return result

def prepare_sd(p,t,period):
    r = open('data/'+str(p)+str(t)+test+'.degap.csv','r')
    r.readline()
    w=[]
    for i in range(0,period):
        d,t,o,h,l,c,a,v = parseLine(r)
        w.append(a)
    result={}
    prev=[]
    while True:
        d,t,o,h,l,c,a,v = parseLine(r)
        if (d==-1): break
        w = w[1:]
        w.append(a)
        sum = 0
        for ww in w:
            sum += ww
        ave = sum/period
        ds = 0
        for ww in w:
            ds += (ww-ave)*(ww-ave)
        value = (ds/period)**0.5
        result[str(d)+' '+t] = [value,prev]
        prev = [value,prev]
    r.close()
    return result

def first_hit(p,t,interval):
    r = open('data/'+str(p)+str(t)+test+'.degap.csv','r')
    r.readline()
    result={}
    prev=[]
    opn=[]
    high=[]
    low=[]
    dt=[]
    while True:
        d,t,o,h,l,c,a,v = parseLine(r)
        if (d==-1): break
        opn.append(o)
        high.append(h)
        low.append(l)
        dt.append(str(d)+' '+t)
    r.close()
    for i in range(0,len(high)):
        o = opn[i]
        value = 0
        for j in range(i,i+3):
            if j==len(high): break
            if high[j]-o >= interval/unit[p] and high[j]-o > o-low[j]:
                value = 1
                result[dt[i]] = [value,prev]
                prev = [value,prev]
                break
            elif o-low[j] >= interval/unit[p]:
                value = -1
                result[dt[i]] = [value,prev]
                prev = [value,prev]
                break
        if value == 0:
            result[dt[i]] = [value,prev]
            prev = [value,prev]
    return result

def prepare_breakout(p,t,period):
    r = open('data/'+str(p)+str(t)+test+'.degap.csv','r')
    r.readline()
    highs=[]
    lows=[]
    for i in range(0, period):
        d,t,o,h,l,c,a,v = parseLine(r)
        highs.append(h)
        lows.append(l)
    result={}
    prev=[]
    length = period - min(int(period*0.2),6)
    while True:
        d,t,o,h,l,c,a,v = parseLine(r)
        if (d==-1): break
        high=max(highs[:length])
        low=min(lows[:length])
        breakout = 0
        if c>high: breakout=c-high
        elif c<low: breakout=c-low
        result[str(d)+' '+t] = [breakout,prev]
        prev = [breakout,prev]
        highs=highs[1:]
        highs.append(h)
        lows=lows[1:]
        lows.append(l)        
    r.close()
    return result
    
def prepare_cci(p,t,period):
    r = open('data/'+str(p)+str(t)+test+'.degap.csv','r')
    r.readline()
    prices=[]
    for i in range(0, period):
        d,t,o,h,l,c,a,v = parseLine(r)
        prices.append(a)
    result={}
    prev=[]
    while True:
        d,t,o,h,l,c,a,v = parseLine(r)
        if (d==-1): break
        prices=prices[1:]
        prices.append(a)
        sma = sum(prices)/period
        md = 0  # mean deviation
        for p in prices: md = md + abs(p-sma)
        md = md / period
        if md==0: cci=0
        else: cci = (a-sma)/(0.015*md)
        result[str(d)+' '+t] = [cci,prev]
        prev = [cci,prev]
    r.close()
    return result
        
def prepare_rsi(p,t,period):
    r = open('data/'+str(p)+str(t)+test+'.degap.csv','r')
    r.readline()
    prices=[]
    d,t,o,h,l,c,aa,v = parseLine(r)    
    for i in range(0, period):
        d,t,o,h,l,c,a,v = parseLine(r)
        prices.append(a-aa)
        aa=a
    ps=0
    ns=0
    for p in prices:
        if p>0: ps=ps+p
        elif p<0: ns=ns-p
    ps=ps/period
    ns=ns/period
    result={}
    prev=[]
    while True:
        d,t,o,h,l,c,a,v = parseLine(r)
        if (d==-1): break
        dif = a-aa
        if dif>=0: 
            ps = (ps*(period-1)+dif)/period
            ns = ns*(period-1)/period
        if dif<=0: 
            ps = ps*(period-1)/period
            ns = (ns*(period-1)-dif)/period
        if ns==0:
            if ps==0: rsi=50
            else: rsi=100
        else:
            rs = ps/ns
            rsi = 100 - 100/(1+rs)
        result[str(d)+' '+t] = [rsi,prev]
        prev = [rsi,prev]
        prices=prices[1:]
        prices.append(a-aa)
        aa=a
    r.close()
    return result

def prepare_atr(p,t,period):
    r = open('data/'+str(p)+str(t)+test+'.degap.csv','r')
    r.readline()
    trange=[]
    d,t,o,h,l,cc,a,v = parseLine(r)
    for i in range(0,period):
        d,t,o,h,l,c,a,v = parseLine(r)
        r1=h-l
        r2=abs(h-cc)
        r3=abs(l-cc)
        tr=max(r1,r2,r3)
        trange.append(tr)
        cc=c        
    result={}
    prev=[]
    while True:
        d,t,o,h,l,c,a,v = parseLine(r)
        if (d==-1): break
        r1=h-l
        r2=abs(h-cc)
        r3=abs(l-cc)
        tr=max(r1,r2,r3)
        trange=trange[1:]
        trange.append(tr)
        last = sum(trange)/period
        result[str(d)+' '+t] = [last,prev]
        prev = [last,prev]
        cc=c
    r.close()
    return result

def prepare_macd(p,tf,period):
    r = open('data/'+str(p)+str(tf)+test+'.degap.csv','r')
    r.readline()
    result={}
    prev=[]
    while True:
        d,t,o,h,l,c,a,v = parseLine(r)
        if (d==-1): break
        last  = indicator(p,"ema",tf,period,d,t,0) - indicator(p,"ema",tf,period*2,d,t,0)
        result[str(d)+' '+t] = [last,prev]
        prev = [last,prev]
    r.close()
    return result

def prepare_macd_signal(pr,tf,period):
    r = open('data/'+str(pr)+str(tf)+test+'.degap.csv','r')
    r.readline()
    p = []
    period2 = int(period*2/3)
    for i in range(0,period2):
        d,t,o,h,l,c,a,v = parseLine(r)
        p.append(indicator(pr,"macd",tf,period,d,t,0))
    result={}
    prev=[]
    while True:
        d,t,o,h,l,c,a,v = parseLine(r)
        if (d==-1): break
        macd=indicator(pr,"macd",tf,period,d,t,0)
        p=p[1:]
        p.append(macd)
        last = sum(p)/period2-macd
        result[str(d)+' '+t] = [last,prev]
        prev = [last,prev]
    r.close()
    return result

def prepare_demarker(p,tf,period):
    r = open('data/'+str(p)+str(tf)+test+'.degap.csv','r')
    r.readline()
    d,t,o,hh,ll,c,a,v = parseLine(r)
    demax = []
    demin = []
    for i in range(0,period):
        d,t,o,h,l,c,a,v = parseLine(r)
        demax.append(max(0,h-hh))
        demin.append(max(0,ll-l))
        hh=h
        ll=l
    result={}
    prev=[]
    while True:
        d,t,o,h,l,c,a,v = parseLine(r)
        if (d==-1): break
        demax=demax[1:]
        demin=demin[1:]
        demax.append(max(0,h-hh))
        demin.append(max(0,ll-l))
        maxma=sum(demax)/period
        minma=sum(demin)/period
        if maxma+minma==0:
            last=0
        else:
            last = maxma / (maxma + minma)
        result[str(d)+' '+t] = [last,prev]
        prev = [last,prev]
        hh=h
        ll=l
    r.close()
    return result

def prepare_force(p,tf,period):
    r = open('data/'+str(p)+str(tf)+test+'.degap.csv','r')
    r.readline()
    for i in range(0,period): d,t,o,h,l,c,a,v = parseLine(r)
    result={}
    prev=[]
    while True:
        d,t,o,h,l,c,a,v = parseLine(r)
        if (d==-1): break
        if d==20190201 and t=="23:50:00": print(v)
        last  = v * (indicator(p,"ema",tf,period,d,t,0) - indicator(p,"ema",tf,period,d,t,1))
        result[str(d)+' '+t] = [last,prev]
        prev = [last,prev]
        cc=c
    r.close()
    return result

def prepare_trix(p,tf,period):
    r = open('data/'+str(p)+str(tf)+test+'.degap.csv','r')
    r.readline()
    d,t,o,h,l,c,a,v = parseLine(r)    
    ee = indicator(p,"ema3",tf,period,d,t,0)
    result={}
    prev=[]
    while True:
        d,t,o,h,l,c,a,v = parseLine(r)
        if (d==-1): break
        e = indicator(p,"ema3",tf,period,d,t,0)
        last  =  (e-ee)/ee
        result[str(d)+' '+t] = [last,prev]
        prev = [last,prev]
        ee=e
    r.close()
    return result

def prepare_adx(p,tf,period):
    r = open('data/'+str(p)+str(tf)+test+'.degap.csv','r')
    r.readline()
    d,t,o,hh,ll,cc,a,v = parseLine(r)
    result={}
    prev=[]
    adx = 0
    pdi = 0
    ndi = 0
    m = 2 / (period+1)
    while True:
        d,t,o,h,l,c,a,v = parseLine(r)
        if (d==-1): break
        trs = max(h-l,abs(h-cc),abs(l-cc))
        hd=h-hh
        ld=ll-l
        if hd<0: hd=0
        if ld<0: ld=0
        if hd>ld: ld=0
        else:
            if hd<ld: hd=0
            else:
                hd=0
                ld=0
        if trs==0:
            pdi=pdi*(1-m)
            ndi=ndi*(1-m)
        else:
            pdi = (hd/trs*100)*m + pdi*(1-m)
            ndi = (ld/trs*100)*m + ndi*(1-m)
        if pdi+ndi==0:
            adx=adx*(1-m)
        else:
            adx = abs((pdi-ndi)/(pdi+ndi))*100*m + adx*(1-m)
        last = [adx,pdi-ndi]
        result[str(d)+' '+t] = [last,prev]
        prev = [last,prev]
        cc=c
        hh=h
        ll=l
    r.close()
    return result
    
def prepare_momentum(p,tf,period):
    r = open('data/'+str(p)+str(tf)+test+'.degap.csv','r')
    r.readline()
    ave=[]
    for i in range(0,period):
        d,t,o,h,l,c,a,v = parseLine(r)
        ave.append(a)
    result={}
    prev=[]
    while True:
        d,t,o,h,l,c,a,v = parseLine(r)
        if (d==-1): break
        last  = a/ave[0]*100
        result[str(d)+' '+t] = [last,prev]
        prev = [last,prev]
        ave=ave[1:]
        ave.append(a)
    r.close()
    return result

def prepare_rvi(p,tf,period):
    r = open('data/'+str(p)+str(tf)+test+'.degap.csv','r')
    r.readline()
    clo=[]
    opn=[]
    hig=[]
    low=[]
    for i in range(0,4):
        d,t,o,h,l,c,a,v = parseLine(r)
        clo.append(c)
        opn.append(o)
        hig.append(h)
        low.append(l)
    rvi=[]
    nume=[]
    deno=[]
    for i in range(0,period):
        d,t,o,h,l,c,a,v = parseLine(r)
        clo=clo[1:]
        opn=opn[1:]
        hig=hig[1:]
        low=low[1:]
        clo.append(c)
        opn.append(o)
        hig.append(h)
        low.append(l)
        nume.append((c-o)+(2*(clo[2]-opn[2]))+(2*(clo[1]-opn[1]))+(clo[0]-opn[0]))
        deno.append((h-l)+(2*(hig[2]-low[2]))+(2*(hig[1]-low[1]))+(hig[0]-low[0]))
        denos = sum(deno)
        if denos==0: rv=sum(nume)
        else: rv=sum(nume)/denos
        if (len(rvi)==4): rvi=rvi[1:]
        rvi.append(rv)        
    result={}
    prev=[]
    while True:
        d,t,o,h,l,c,a,v = parseLine(r)
        if (d==-1): break
        clo=clo[1:]
        opn=opn[1:]
        hig=hig[1:]
        low=low[1:]
        clo.append(c)
        opn.append(o)
        hig.append(h)
        low.append(l)
        nume=nume[1:]
        deno=deno[1:]
        nume.append((c-o)+(2*(clo[2]-opn[2]))+(2*(clo[1]-opn[1]))+(clo[0]-opn[0]))
        deno.append((h-l)+(2*(hig[2]-low[2]))+(2*(hig[1]-low[1]))+(hig[0]-low[0]))
        denos = sum(deno)
        if denos==0: rv=sum(nume)
        else: rv=sum(nume)/denos
        rvi=rvi[1:]
        rvi.append(rv)
        signal=(rv+(2*rvi[2])+(2*rvi[1])+rvi[0])/6
        last  = [signal-rv,rv]
        result[str(d)+' '+t] = [last,prev]
        prev = [last,prev]
    r.close()
    return result

def prepare_strength(p,tf,period):
    pd = {}   # pair difference
    for p in pairs99:
        pd[p] = {}
        r = open('data/'+str(p)+str(tf)+test+'.degap.csv','r')
        r.readline()
        while True:
            d,t,o,h,l,c,a,v = parseLine(r)
            if (d==-1): break
            try:
                pd[p][str(d)+' '+t] = c/o
            except:
                print("error: "+str(o))
                pd[p][str(d)+' '+t] = 0
        r.close()
    result={}
    prev=[]
    r = open('data/GBPJPY'+str(tf)+test+'.degap.csv','r')
    r.readline()
    cs = {}   # currency strength
    csi = {}  # iterations
    while True:
        d,t,o,h,l,c,a,v = parseLine(r)
        if (d==-1): break
        cs["USD"] = 1
        if (str(d)+' '+t) in pd["GBPUSD"]: cs["GBP"] = pd["GBPUSD"][str(d)+' '+t]
        else: cs["GBP"] = 1
        if (str(d)+' '+t) in pd["EURUSD"]: cs["EUR"] = pd["EURUSD"][str(d)+' '+t]
        else: cs["EUR"] = 1
        if (str(d)+' '+t) in pd["NZDUSD"]: cs["NZD"] = pd["NZDUSD"][str(d)+' '+t]
        else: cs["NZD"] = 1
        if (str(d)+' '+t) in pd["AUDUSD"]: cs["AUD"] = pd["AUDUSD"][str(d)+' '+t]
        else: cs["AUD"] = 1
        if (str(d)+' '+t) in pd["USDJPY"]: cs["JPY"] = 1 / pd["USDJPY"][str(d)+' '+t]
        else: cs["JPY"] = 1
        if (str(d)+' '+t) in pd["USDCHF"]: cs["CHF"] = 1 / pd["USDCHF"][str(d)+' '+t]
        else: cs["CHF"] = 1
        if (str(d)+' '+t) in pd["USDCAD"]: cs["CAD"] = 1 / pd["USDCAD"][str(d)+' '+t]
        else: cs["CAD"] = 1
        last = deepcopy(cs)
        result[str(d)+' '+t] = [last,prev]
        prev = [last,prev]
    r.close()
    return result

def prepare_bb(p,tf,period):
    r = open('data/'+str(p)+str(tf)+test+'.degap.csv','r')
    r.readline()
    result={}
    prev=[]
    cc = []
    for i in range(period):
        d,t,o,h,l,c,a,v = parseLine(r)
        cc.append(c)
    while True:
        d,t,o,h,l,c,a,v = parseLine(r)
        if (d==-1): break
        cc = cc[1:]
        cc.append(c)
        sma = sum(cc) / period
        sd = indicator(p,"sd",tf,period,d,t,0)
        if abs(sd)>0 :last  = (c - sma) / 2 / sd
        else: last=0
        result[str(d)+' '+t] = [last,prev]
        prev = [last,prev]
    r.close()
    return result

def prepare_psar(p,tf,period):
    result={}
    prev=[]
    r = open('data/'+str(p)+str(tf)+test+'.degap.csv','r')
    r.readline()
    hh = []
    ll = []
    ep = 0
    pph = 0
    ppl = 0
    ph = 0
    pl = 0
    af = 0.02
    for i in range(20):
        d,t,o,h,l,c,a,v = parseLine(r)
        if l < ep: ep = l   # extremum point
        pph=ph              # pre-previous h
        ppl=pl              # pre-previous l
        ph = h              # previous high        
        pl = l              # previous low
        hh.append(h)
        ll.append(l)
    psar = ep
    isLong = False
    while True:
        d,t,o,h,l,c,a,v = parseLine(r)
        if (d==-1): break
        if isLong:
            if h > ep:
                if af<0.2: af = af + 0.02
                ep = h
            psar = psar + af * (ep-psar)
            if psar > pl : psar = pl
            if psar > ppl: psar = ppl
            if psar>l:
                isLong = False
                psar = ep
                ep = min(ll)
                hh = [h]
                af = 0.02    # acceleration factor
            ll.append(l)
        else:
            if l < ep:
                if af<0.2: af = af + 0.02
                ep = l
            psar = psar + af * (ep-psar)
            if psar < ph : psar = ph
            if psar < pph: psar = pph
            if psar<h:
                isLong = True
                psar = ep
                ep = max(hh)
                ll = [l]
                alpha = 0.02
            hh.append(h)
        pph=ph
        ppl=pl
        ph = h
        pl = l
        last  = c - psar
        result[str(d)+' '+t] = [last,prev]
        prev = [last,prev]
    r.close()
    return result

def prepare_lookahead(p,t):
    r = open('data/'+str(p)+str(t)+test+'.degap.csv','r')
    r.readline()
    result={}
    prev=[]
    opn=[]
    close=[]
    dt=[]
    if p=="USATECHIDXUSD": divisor = 1000
    else: divisor = 1
    while True:
        d,t,o,h,l,c,a,v = parseLine(r,divisor)
        if (d==-1): break
        opn.append(o)
        close.append(c)
        dt.append(str(d)+' '+t)
    r.close()
    for i in range(len(close)-1,-1,-1):
        value = [opn[i],close[i]]
        result[dt[i]] = [value,prev]
        prev = [value,prev]
    return result


def prepare_ideal_lot(p,t):
    r = open('data/'+str(p)+str(t)+test+'.degap.csv','r')
    r.readline()
    result={}
    prev=[]
    opn=[]
    close=[]
    lo=[]
    hi=[]
    dt=[]
    while True:
        d,t,o,h,l,c,a,v = parseLine(r)
        if (d==-1): break
        opn.append(o)
        hi.append(h)
        lo.append(l)
        close.append(c)
        dt.append(str(d)+' '+t)
    r.close()
    sp = commission[p]/unit[p] + spread[p]
    for i in range(len(close)):
        net=0
        j=i
        while net>-sp and net<sp:
            net = net + close[j] - opn[j]
            j=j+1
            if j==len(close):
                v=0
                break
        if net>0 and j<len(close):
            net=0
            mx=-999999999
            j=i
            while net>-sp:
                if close[j]>mx:
                    mx=close[j]
                    k=j
                    net=0
                else:
                    net = net + close[j] - opn[j]
                j=j+1
                if j==len(close):
                    v=0
                    break
            if j<len(close):
                mn=999999999
                j=i
                while (j<=k):
                    if lo[j]<mn: mn=lo[j]
                    j=j+1
                if (opn[i]-mn)==0: v = 3
                else: v = 100/(((opn[i]-mn) + spread[p]) * unit[p] / value[p[-3:]] + commission[p])
        elif net<0 and j<len(close):
            net=0
            mn=999999999
            j=i
            while net<sp:
                if close[j]<mn:
                    mn=close[j]
                    k=j
                    net=0
                else:
                    net = net + close[j] - opn[j]
                j=j+1
                if j==len(close):
                    v=0
                    break
            if j<len(close):
                mx=0
                j=i
                while (j<=k):
                    if hi[j]>mx: mx=hi[j]
                    j=j+1
                if (opn[i]-mx)==0: v = -3
                else: v = 100/(((opn[i]-mx) - spread[p]) * unit[p] / value[p[-3:]] - commission[p])
        result[dt[i]] = [v,prev]
        prev = [v,prev]
    return result

def prepare_abcd(p,t,period):
    r = open('data/'+str(p)+str(t)+test+'.degap.csv','r')
    r.readline()
    result={}
    prev=[]
    oo = []
    hh = []
    ll = []
    cc = []
    dt=[]
    while True:
        d,t,o,h,l,c,a,v = parseLine(r)
        if (d==-1): break
        oo.append(o)
        hh.append(h)
        ll.append(l)
        cc.append(c)
        dt.append(str(d)+' '+t)
    r.close()
    for i in range(len(dt)-period):
        minv = ll[i]
        maxv = hh[i]
        mini = i
        maxi = i
        for j in range(period):
            if j==0: continue
            if ll[i+j]<minv:
                minv = ll[i+j]
                mini = i+j
            if hh[i+j]>maxv:
                maxv = hh[i+j]
                maxi = i+j
        o = oo[i]
        if mini < maxi:
            a = minv - o
            b = maxv - minv
            c = maxv - o
            d = cc[i+period] - o
        else:
            a = maxv - o
            b = minv - maxv
            c = minv - o
            d = cc[i+period] - o
        last  = [a,b,c,d]
        result[dt[i]] = [last,prev]
        prev = [last,prev]
    return result

def test2(p,tf,period,d,t,shift=0):
    print("Testing indidcators at "+str(d)+"  "+t+"--------------------------------------")
    print("ema: "+str(indicator(p,"ema",tf,period,d,t,0)))
    print("rvi dif: "+str(indicator(p,"rvi",tf,period,d,t,0)[0]))
    print("rv: "+str(indicator(p,"rvi",tf,period,d,t,0)[1]))