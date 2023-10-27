import pickle
import numpy as np
import csv,os
import sched, time
from datetime import datetime

curr  = ["USD","JPY","EUR","GBP","AUD","CAD","CHF","NZD"]
curr2 = [      "JPY","EUR","GBP","AUD","CAD","CHF","NZD"]
curr3 = [            "EUR","GBP","AUD","CAD","CHF","NZD"]
pairs0 = ["EURUSD","GBPUSD","USDCAD","USDJPY","USDCHF","AUDUSD","NZDUSD","EURJPY","GBPJPY","CADJPY","CHFJPY","AUDJPY","NZDJPY","EURGBP","EURCAD","EURCHF","EURAUD","EURNZD","GBPCAD","GBPCHF","GBPAUD","GBPNZD","CADCHF","AUDCAD","NZDCAD","AUDCHF","NZDCHF","AUDNZD","USDSGD","XAUUSD","LIGHTCMDUSD","AUDSGD", "CHFSGD", "EURNOK", "EURPLN","EURSEK","EURSGD","EURTRY","EURZAR","SGDJPY","USDMXN","USDNOK","USDPLN","USDRUB","USDSEK","USDTRY","USDZAR","USDHUF","EURHUF","EURMXN","ZARJPY","USDCNH","XAGUSD","USA30IDXUSD","USATECHIDXUSD","USA500IDXUSD","PDCMDUSD","PTCMDUSD","CUCMDUSD"]
pairs1 = ["EURUSD","GBPUSD","USDCAD","USDJPY","USDCHF","AUDUSD","NZDUSD"]
pairs2 = ["EURJPY","GBPJPY","CADJPY","CHFJPY","AUDJPY","NZDJPY","EURGBP"]
pairs3 = ["EURCAD","EURCHF","EURAUD","EURNZD","GBPCAD","GBPCHF","GBPAUD"]
pairs4 = ["GBPNZD","CADCHF","AUDCAD","NZDCAD","AUDCHF","NZDCHF","AUDNZD"]

pairs5 = ["GBPJPY","GBPAUD","GBPNZD","AUDJPY"]
pairs91 = ["AUDSGD", "CHFSGD", "EURNOK", "EURPLN", "EURSEK", "EURSGD", "EURTRY", "EURZAR", "SGDJPY", "USDMXN", "USDNOK", "USDPLN", "USDRUB", "USDSEK", "USDTRY", "USDZAR", "ZARJPY", "USDCNH", "XAGUSD", "USA30IDXUSD", "USATECHIDXUSD", "USA500IDXUSD", "PDCMDUSD", "PTCMDUSD", "CUCMDUSD"]

pairs99 = ["EURUSD","GBPUSD","USDCAD","USDJPY","USDCHF","AUDUSD","NZDUSD","EURJPY","GBPJPY","CADJPY","CHFJPY","AUDJPY","NZDJPY","EURGBP","EURCAD","EURCHF","EURAUD","EURNZD","GBPCAD","GBPCHF","GBPAUD","GBPNZD","CADCHF","AUDCAD","NZDCAD","AUDCHF","NZDCHF","AUDNZD"]
pairs9 = ["LIGHTCMDUSD"]
old = ["GBPJPY"]

exotics = ["USDCNH","USDHUF","USDMXN","USDPLN","USDRUB","USDSGD","USDTRY","USDZAR","USDSEK","USDNOK","USATECHIDXUSD","LIGHTCMDUSD"]
exotics2 = ["USDCNH","USDHUF","USDMXN","USDNOK","USDPLN","USDRUB","USDSEK","USDSGD","USDTRY","USDZAR","EURHUF","EURMXN","EURNOK","EURPLN","EURSEK","EURSGD","EURTRY","EURZAR","AUDSGD","CHFSGD","SGDJPY","ZARJPY"]
commo  = ["XAUUSD","XTIUSD","XAGUSD"]
indices = ["US30","US500","NAS100","AUS200","EUSTX50","FRA40","GER30","HK50","IT40","JPN225","UK100","SPA35"]
stocks = ['US500','MMM','T','ABT','ADBE','AA','AAPL','AMZN','BA','BABA','GOOGL','MO','AMGN','BAC','BMY','AVGO','CAT','CVX','CSCO','C','CMCSA','FB','XOM','GE','GOOG','GS','HON','IBM','INTC','JPM','JNJ','MA','MCD','MRK','MSFT','NKE','NVDA','NFLX','ORCL','PEP','PFE','PM','PG','SPY','TXN','KO','HD','DIS','UNP','UNH','VZ','V','WMT','WFC','TSLA','TWTR','QQQ','XLF','EEM']

chosen = ["XAUUSD","AUDSGD", "CHFSGD", "EURNOK", "EURPLN", "EURSEK", "EURSGD", "USDMXN", "USDNOK", "USDPLN", "USDSEK", "USDZAR","USDSGD"]
chosen2 = ["AUDSGD", "CHFSGD", "EURNOK", "EURPLN", "EURSEK", "EURSGD", "USDMXN", "USDNOK", "USDPLN", "USDSEK", "USDZAR","USA30IDXUSD", "USA500IDXUSD","XAUUSD","LIGHTCMDUSD","USDHUF","EURHUF","EURMXN" ]
chosen3 = ["AUDSGD", "CHFSGD", "EURNOK", "EURPLN", "EURSEK", "EURSGD", "USDMXN", "USDNOK", "USDPLN", "USDSEK", "USDZAR"]

threshold = 0
factor = 0.5
set_limit = 28
cutoff_adjustor = 3e-10
kn_size = 300
#test = "_test"
test = ""

def fdp(f):
    return '{:.7f}'.format(f)

def fsn(f):
    if (abs(f)>0.0001): return str(round(f,4))
    s = str(f)
    left=""
    if f<0: left = s[0:4]
    else: left = s[0:3]
    if left[len(left)-1]=="-": left=left[0:len(left)-2]
    right=s
    while len(right)>1 and right[0]!="e": right=right[1:]
    return left+right

last_close = {}

commission = {
    "EURUSD": 7,
    "GBPUSD": 7,
    "EURJPY": 0.7,
    "AUDUSD": 7,
    "GBPJPY": 0.7,
    "USDCAD": 7,
    "USDJPY": 0.7,
    "USDCHF": 7,
    "NZDUSD": 7,
    "CADJPY": 0.7,
    "CHFJPY": 0.7,
    "AUDJPY": 0.7,
    "NZDJPY": 0.7,
    "EURGBP": 7,
    "EURCAD": 7,
    "EURCHF": 7,
    "EURAUD": 7,
    "EURNZD": 7,
    "GBPCAD": 7,
    "GBPCHF": 7,
    "GBPAUD": 7,
    "GBPNZD": 7,
    "CADCHF": 7,
    "AUDCAD": 7,
    "NZDCAD": 7,
    "AUDCHF": 7,
    "NZDCHF": 7,
    "AUDNZD": 7,
    "USDSGD": 7,

    "AUDSGD": 7,
    "CHFSGD": 7,
    "EURHUF": 7,
    "EURMXN": 7,
    "EURNOK": 7,
    "EURPLN": 7,
    "EURSEK": 7,
    "EURSGD": 7,
    "EURTRY": 7,
    "EURZAR": 7,
    "SGDJPY": 7,
    "USDCNH": 7,
    "USDHKD": 7,    
    "USDHUF": 7,
    "USDMXN": 7,
    "USDNOK": 7,
    "USDPLN": 7,
    "USDRUB": 7,
    "USDSEK": 7,
    "USDTRY": 7,
    "USDZAR": 7,
    "ZARJPY": 7,

    "USA30IDXUSD": 0,
    "USA500IDXUSD": 0,
    "NAS100": 0,
    "US30":0,
    "US500":0,
    "NAS100":0,
    "USATECHIDXUSD":0,
    "AUS200":0,
    "EUSTX50":0,
    "FRA40":0,
    "GER30":0,
    "HK50":0,
    "IT40":0,
    "JPN225":0,
    "UK100":0,
    "SPA35":0,

    "XAUUSD": 0,
    "XAGUSD": 0,
    "PDCMDUSD": 0,
    "PTCMDUSD": 0,
    "CUCMDUSD": 0,
    "XTIUSD": 0,
    "LIGHTCMDUSD": 0
}

spread = {
    "EURUSD": 0.00001,
    "GBPUSD": 0.00002,
    "EURJPY": 0.002,
    "AUDUSD": 0.00002,
    "GBPJPY": 0.016,
    "USDCAD": 0.00003,
    "USDJPY": 0.001,
    "USDCHF": 0.00003,
    "NZDUSD": 0.00002,
    "CADJPY": 0.0035,
    "CHFJPY": 0.003,
    "AUDJPY": 0.003,
    "NZDJPY": 0.006,
    "EURGBP": 0.00002,
    "EURCAD": 0.00009,
    "EURCHF": 0.000025,
    "EURAUD": 0.00007,
    "EURNZD": 0.00016,
    "GBPCAD": 0.00017,
    "GBPCHF": 0.00014,
    "GBPAUD": 0.00025,
    "GBPNZD": 0.00035,
    "CADCHF": 0.00009,
    "AUDCAD": 0.00007,
    "NZDCAD": 0.00023,
    "AUDCHF": 0.00008,
    "NZDCHF": 0.00014,
    "AUDNZD": 0.00011,
    "USDSGD": 0.00025,
    "XAUUSD": 2500,

    "AUDSGD": 0.00029,
    "CHFSGD": 0.00022,
    "EURHUF": 0.090,
    "EURMXN": 0.00900,
    "EURNOK": 0.00220,
    "EURPLN": 0.00120,
    "EURSEK": 0.00200,
    "EURSGD": 0.00020,
    "EURTRY": 0.00600,
    "EURZAR": 0.00800,
    "SGDJPY": 0.010,
    "USDHUF": 0.120,
    "USDMXN": 0.00400,
    "USDNOK": 0.00150,
    "USDPLN": 0.00100,
    "USDRUB": 0.01500,
    "USDSEK": 0.00180,
    "USDTRY": 0.00600,
    "USDZAR": 0.00900,
    "ZARJPY": 0.010,

    "USA30IDXUSD": 3000,
    "NAS100": 1.0,    
    "USA500IDXUSD": 800,
    "US30":2.5,
    "US500":0.6,
    "NAS100":1,
    "USATECHIDXUSD":1.9,
    "AUS200":3.8,
    "EUSTX50":2,
    "FRA40":1,
    "GER30":1,
    "HK50":30,
    "IT40":10,
    "JPN225":8,
    "UK100":1,
    "SPA35":5,

    "USDCNH": 0.0007,
    "XAGUSD": 0.012,
    "PDCMDUSD": 2.0,
    "PTCMDUSD": 1.5,
    "CUCMDUSD": 0.0051,
    "XTIUSD": 0.07,
    "LIGHTCMDUSD": 0.07
}

value = {
    "USD":1,
    "EUR":0.92,
    "JPY":109,
    "GBP":0.77,
    "CAD":1.327,
    "AUD":1.49,
    "NZD":1.55,
    "CHF":0.99,
    "CNH":7.1,
    "HUF":308,
    "MXN":20,
    "NOK":9.1,
    "PLN":4,
    "RUB":64.4,
    "SEK":9.8,
    "SGD":1.38,
    "TRY":5.68,
    "ZAR":15,
    "S30":1,
    "500":1,
    "NAS":1,    
    "200":1.48,
    "X50":0.92,
    "A40":0.92,
    "R30":0.92,
    "K50":7.84,
    "T40":0.92,
    "225":108,
    "100":0.81,
    "A35":0.92
}


practical_name = {
    "EURUSD": "EURUSD",
    "GBPUSD": "GBPUSD",
    "EURJPY": "EURJPY",
    "AUDUSD": "AUDUSD",
    "GBPJPY": "GBPJPY",
    "USDCAD": "USDCAD",
    "USDJPY": "USDJPY",
    "USDCHF": "USDCHF",
    "NZDUSD": "NZDUSD",
    "CADJPY": "CADJPY",
    "CHFJPY": "CHFJPY",
    "AUDJPY": "AUDJPY",
    "NZDJPY": "NZDJPY",
    "EURGBP": "EURGBP",
    "EURCAD": "EURCAD",
    "EURCHF": "EURCHF",
    "EURAUD": "EURAUD",
    "EURNZD": "EURNZD",
    "GBPCAD": "GBPCAD",
    "GBPCHF": "GBPCHF",
    "GBPAUD": "GBPAUD",
    "GBPNZD": "GBPNZD",
    "CADCHF": "CADCHF",
    "AUDCAD": "AUDCAD",
    "NZDCAD": "NZDCAD",
    "AUDCHF": "AUDCHF",
    "NZDCHF": "NZDCHF",
    "AUDNZD": "AUDNZD",
    "USDSGD": "USDSGD",
    "XAUUSD": "XAUUSD",

    "AUDSGD":"AUDSGD",
    "CHFSGD":"CHFSGD",
    "EURHUF":"EURHUF",
    "EURMXN":"EURMXN",
    "EURNOK":"EURNOK",
    "EURPLN":"EURPLN",
    "EURSEK":"EURSEK",
    "EURSGD":"EURSGD",
    "EURTRY":"EURTRY",
    "EURZAR":"EURZAR",
    "SGDJPY":"SGDJPY",
    "USDHUF":"USDHUF",
    "USDMXN":"USDMXN",
    "USDNOK":"USDNOK",
    "USDPLN":"USDPLN",
    "USDRUB":"USDRUB",
    "USDSEK":"USDSEK",
    "USDTRY":"USDTRY",
    "USDZAR":"USDZAR",
    "ZARJPY":"ZARJPY",

    "USA30IDXUSD": "US30",
    "USA500IDXUSD": "US500",
    "USATECHIDXUSD": "NAS100",

    "USDCNH":"USDCNH",
    "XAGUSD":"XAGUSD",
    "PDCMDUSD":"XPDUSD",
    "PTCMDUSD":"XPTUSD",
    "CUCMDUSD":"Copper",
    "LIGHTCMDUSD":"XTIUSD"
}

unit = {
    "EURUSD": 100000,
    "GBPUSD": 100000,
    "EURJPY": 1000,
    "AUDUSD": 100000,
    "GBPJPY": 1000,
    "USDCAD": 100000,
    "USDJPY": 1000,
    "USDCHF": 100000,
    "NZDUSD": 100000,
    "CADJPY": 1000,
    "CHFJPY": 1000,
    "AUDJPY": 1000,
    "NZDJPY": 1000,
    "EURGBP": 100000,
    "EURCAD": 100000,
    "EURCHF": 100000,
    "EURAUD": 100000,
    "EURNZD": 100000,
    "GBPCAD": 100000,
    "GBPCHF": 100000,
    "GBPAUD": 100000,
    "GBPNZD": 100000,
    "CADCHF": 100000,
    "AUDCAD": 100000,
    "NZDCAD": 100000,
    "AUDCHF": 100000,
    "NZDCHF": 100000,
    "AUDNZD": 100000,
    "USDSGD": 100000,
    "XAUUSD": 1,

    "AUDSGD":100000,
    "CHFSGD":100000,
    "EURHUF":100000,
    "EURMXN":100000,
    "EURNOK":100000,
    "EURPLN":100000,
    "EURSEK":100000,
    "EURSGD":100000,
    "EURTRY":100000,
    "EURZAR":100000,
    "SGDJPY":1000,
    "USDHUF":100000,    
    "USDMXN":100000,
    "USDNOK":100000,
    "USDPLN":100000,
    "USDRUB":100000,
    "USDSEK":100000,
    "USDTRY":100000,
    "USDZAR":100000,
    "ZARJPY":1000,

    "USA30IDXUSD": 0.001,
    "USA500IDXUSD": 0.001,
    "NAS100": 1,
    "USATECHIDXUSD":1,
    "US30":1,
    "US500":1,
    "NAS100":1,
    "AUS200":1,
    "EUSTX50":1,
    "FRA40":1,
    "GER30":1,
    "HK50":1,
    "IT40":1,
    "JPN225":1,
    "UK100":1,
    "SPA35":1,

    "USDCNH": 100000,
    "USDHKD": 100000,
    "XAGUSD": 5000,
    "PDCMDUSD": 100,
    "PTCMDUSD": 100,
    "CUCMDUSD": 2000,
    "XTIUSD": 100,
    "LIGHTCMDUSD": 100
}

sortKey = 0
def getSortKey(item):
    global sortKey
    return item[sortKey]
def normalize(dat,f,positiveResult=False,isEnsemble=False,isClassifier=False):
    global sortKey
    d = []
    lf = []
    factors=[]
    yo=[]        # original y
    last = len(dat[0])
    for dd in dat: d.append(dd[:])
    for i in  range(0,last):
        if i==287: continue    # date
        print("normalizing key "+str(i))
        sortKey=i
        d.sort(reverse=True,key=getSortKey)
        l = len(d)
        ff = f
        if i>=14 and ((i-14)%16==4) and i<last-14: ff = 0.025
        tf = abs(d[round(ff/100*l)][i])
        bf = abs(d[round((1-ff/100)*l)][i])
        if positiveResult: sf = (abs(d[round(ff/200*l)][i])+abs(d[round((1-ff/200)*l)][i]))
        else: sf = (abs(d[round(ff/200*l)][i])+abs(d[round((1-ff/200)*l)][i]))/2
        for j in range(0,len(d)):            
            if i == last - 12: yo.append(dat[j][i])
            if i == 0:
                dat[j][i] /= -bf
                if j==0:
                    factors.append(-bf)
            elif i == 1 or (isEnsemble and i==set_limit+1):
                dat[j][i] /= tf
                if j==0:
                    factors.append(tf)
            elif i >= 14 and (i - 14) % 16 == 2 and i<last-14 and not (isEnsemble and i>set_limit):
                dat[j][i] /= -100
                if j==0:
                    factors.append(-100)
            elif i >= 14 and (((i - 14) % 16 == 3) or ((i - 14) % 16 == 6) or ((i - 14) % 16 == 7) or ((i - 14) % 16 == 10) or ((i - 14) % 16 == 12) or ((i - 14) % 16 == 14)) and i<last-14 and not (isEnsemble and i>set_limit):
                dat[j][i] /= tf
                if j==0:
                    factors.append(tf)
            elif i == last-14:
                dat[j][i] /= 24
                if j==0 :
                    factors.append(24)
            else:
                if sf != 0:
                    if positiveResult: dat[j][i] = (dat[j][i] / sf)+0.5
                    else: dat[j][i] = dat[j][i] / sf 
                if j==0 and i<last-14:
                    factors.append(sf)
            if positiveResult: dat[j][i] = max(0,min(1,dat[j][i]))
            else: dat[j][i] = max(-1,min(1,dat[j][i]))
            if j==0 and i >= last - 12: lf.append(sf)
            #if i<3 and j==0:
            #    if j==0:
            #        print(factors)
            #        print(str(bf)+":"+str(tf)+":"+str(sf))
            #    for z in range(0,240000):
            #        if z % 1000 > 0: continue
            #        print(str(i)+ " " +str(z)+" *** "+str(d[z][i]))
    #print("filtering by Isolation Forest...")
    #isof = IsolationForest(contamination=0.02,behaviour='new',n_jobs=5,verbose=2,n_estimators=50)
    #isr = isof.fit_predict(np.array(dat).copy())
    x=[]
    y=[]
    z=[]
    for i in range(0,len(dat)):
        if isClassifier:
            x.append(dat[i][:last-12])
            y.append(dat[i][last-12:])
            z.append(yo[i])
        else:
            x.append(dat[i][:last-12])
            y.append(dat[i][last-12:])
            z.append(yo[i])
    return factors,lf,x,y,z

def normalizeKN(dat,f):
    global sortKey
    d = []
    lf = []
    factors=[]
    yo=[]        # original y
    last = len(dat[0])
    for dd in dat:
        d.append(dd)
    for i in  range(0,last):
        if i==last-2: continue    # date
        sortKey=i
        d.sort(reverse=True,key=getSortKey)
        #print(dat[0])
        l = len(d)
        sf = (abs(d[round(f/200*l)][i])+abs(d[round((1-f/200)*l)][i]))/2
        print("normalizing key "+str(i)+"...eg.:"+str(dat[0][i])+"...eg.:"+str(fdp(sf)))

        for j in range(len(d)):            
            if i == last-1: yo.append(dat[j][i])
            if i == last-3:
                dat[j][i] /= 24
                if j==0 :
                    factors.append(24)
            else:
                if sf != 0: dat[j][i] = dat[j][i] / sf 
                if j==0 and i<last-1: factors.append(sf)
            dat[j][i] = max(-1,min(1,dat[j][i]))
            if j==0 and i == last - 1: lf.append(sf)
    x=[]
    y=[]
    z=[]
    for i in range(len(dat)):
        x.append(dat[i])
        y.append(dat[i])
        z.append(yo[i])
    return factors,lf,x,y,z


def initialize_last_close(pairs):
    global last_close
    for p in pairs:
        r = open('data/'+str(p)+'240.csv','r')
        r.readline()
        lc = 1
        while True:
            d,t,o,h,l,c,a,v = parseLine(r)
            if (d==-1): break
            lc = c
        r.close()
        last_close[p] = lc
        print(p + " last close price: " + str(lc))

def parseLine(r):
    l = r.readline()
    if not l: return -1,-1,-1,-1,-1,-1,-1,-1
    s = l.split(',')
    o = float(s[2])
    h = float(s[3])
    l = float(s[4])
    c = float(s[5])
    e = s[6].find('\n')==-1
    return int(s[0]),s[1],o,h,l,c,(h+l+c*2)/4,float(s[6])

def readNextTick(f):
    def intToBin(n, l):
        r=""
        if (n == 0): r="0"
        else: 
            while n!=0:
                r = str(n % 2)+r;  
                n=int(n/2)
        rl = l-len(r)
        for i in range(0,rl): r="0"+r
        return r
    def binToInt(s,start,length):
        s = s[start:start+length]
        n=0
        for j in range(0,length):
            v=0
            if (s[j]=="1"): v = 1
            n+=pow(2,length-1-j)*v
        return n

    tick = f.read(8)
    if len(tick)<8: return -1,-1,-1,-1,-1,-1,-1,-1
    binary = intToBin(tick[0],8)+intToBin(tick[1],8)+intToBin(tick[2],8)+intToBin(tick[3],8)+intToBin(tick[4],8)+intToBin(tick[5],8)+intToBin(tick[6],8)+intToBin(tick[7],8)
    if len(binary)<64: print("ERROR-------------------------------------------------------------------")
    y = binToInt(binary,0,5)
    m = binToInt(binary,5,4)
    d = binToInt(binary,9,5)
    h = binToInt(binary,14,5)
    n = binToInt(binary,19,6)
    s = binToInt(binary,25,6)
    dp= binToInt(binary,31,3)
    b = binToInt(binary,34,30)
    return y,m,d,h,n,s,b/pow(10,dp),tick

def z(n):
    if int(n)>9: return str(n)
    return "0"+str(n)

def generate(p,tf,start=201812):
    w = open("ticks/"+p+str(tf)+".csv","w")
    firstOpen = True
    firstReach = False
    oldH = -1
    oldM = -1
    lo = 999999
    hi = -999999
    v = 0
    for year in range(int(start/100),2020):
        for mon in range(start%100,13):
            try: r = open('ticks/merged/'+str(year)+'-'+str(mon)+'/'+p+'.tick', "rb")
            except IOError: continue
            print("reading "+str(year)+"-"+str(mon)+" ......")
            while True:
                y,m,d,h,n,s,b,t = readNextTick(r)
                if y<0: break
                if firstOpen: 
                    o = b
                    firstOpen = False
                if ((oldH<23 and abs(h-oldH)>1) or (n%tf < oldM%tf) or (tf==1 and oldM!=n)) and firstReach:
                    w.write(str(y+2000)+z(m)+z(d)+","+z(h)+":"+z(n)+":"+z(s)+","+str(o)+","+str(hi)+","+str(lo)+","+str(c)+","+str(v)+"\n")
                    o = b
                    lo = b
                    hi = b
                    v = 1
                    firstReach=False
                c = b
                if b<lo: lo = b
                if b>hi: hi = b
                v += 1
                if (oldM!=n and firstReach==False): firstReach=True
                oldH = h
                oldM = n
            r.close()
    w.close()

def mergeT(p,tf,start=201812):
    for year in range(int(start/100),2020):
        for mon in range(start%100,13):
            dir = "ticks/merged/"+str(year)+"-"+str(mon)+"/"
            if not os.path.exists(dir): os.makedirs(dir)
            w = open(dir+p+".tick","wb")
            flags = 3
            try: r1 = open('ticks/server/'+str(year)+'-'+str(mon)+'/'+p+'.tick', "rb")
            except IOError: 
                print("skipping ticks/server/"+str(year)+'-'+str(mon)+'/'+p+'.tick')
                flags -= 1
            try: r2 = open('ticks/client/'+str(year)+'-'+str(mon)+'/'+p+'.tick', "rb")
            except IOError: 
                print("skipping ticks/client/"+str(year)+'-'+str(mon)+'/'+p+'.tick')
                flags -= 2
            if flags==0: continue
            print("reading "+str(year)+"-"+str(mon)+" ......")
            if flags==1:
                while True:
                    y,m,d,h,n,s,b,t = readNextTick(r1)
                    if y<0: break
                    w.write(t)
                r1.close()
            elif flags==2:
                while True:
                    y,m,d,h,n,s,b,t = readNextTick(r2)
                    if y<0: break
                    w.write(t)
                r2.close()
            elif flags==3:
                while True:
                    y1,m1,d1,h1,n1,s1,b1,t1 = readNextTick(r1)
                    y2,m2,d2,h2,n2,s2,b2,t2 = readNextTick(r2)
                    
                    if y1<0 and y2<0: break
                    elif y1<0:
                        while True:
                            w.write(t2)
                            print("gap server at "+str(y2)+"-"+str(m2)+"-"+str(d2)+" "+str(h2)+":"+str(n2)+":"+str(s2))
                            y2,m2,d2,h2,n2,s2,b2,t2 = readNextTick(r2)
                            if y2<0: break
                        break
                    elif y2<0:
                        while True:
                            w.write(t1)
                            print("gap client at "+str(y1)+"-"+str(m1)+"-"+str(d1)+" "+str(h1)+":"+str(n1)+":"+str(s1))
                            y1,m1,d1,h1,n1,s1,b1,t1 = readNextTick(r1)
                            if y1<0: break
                        break
                    else:
                        while d1<d2 or (d1==d2 and h1<h2) or (d1==d2 and h1==h2 and n1<n2) or (d1==d2 and h1==h2 and n1==n2 and s1<s2):
                            w.write(t1)
                            print("gap client at "+str(y1)+"-"+str(m1)+"-"+str(d1)+" "+str(h1)+":"+str(n1)+":"+str(s1))
                            y1,m1,d1,h1,n1,s1,b1,t1 = readNextTick(r1)
                            if y1<0: break
                        while d1>d2 or (d1==d2 and h1>h2) or (d1==d2 and h1==h2 and n1>n2) or (d1==d2 and h1==h2 and n1==n2 and s1>s2):
                            w.write(t2)
                            print("gap server at "+str(y2)+"-"+str(m2)+"-"+str(d2)+" "+str(h2)+":"+str(n2)+":"+str(s2))
                            y2,m2,d2,h2,n2,s2,b2,t2 = readNextTick(r2)
                            if y2<0: break
                    
                    if y1<0 and y2<0: break
                    elif y1<0:
                        while True:
                            w.write(t2)
                            print("gap server at "+str(y2)+"-"+str(m2)+"-"+str(d2)+" "+str(h2)+":"+str(n2)+":"+str(s2))
                            y2,m2,d2,h2,n2,s2,b2,t2 = readNextTick(r2)
                            if y2<0: break
                        break
                    elif y2<0:
                        while True:
                            w.write(t1)
                            print("gap client at "+str(y1)+"-"+str(m1)+"-"+str(d1)+" "+str(h1)+":"+str(n1)+":"+str(s1))
                            y1,m1,d1,h1,n1,s1,b1,t1 = readNextTick(r1)
                            if y1<0: break
                        break
                    else:
                        w.write(t1)
                r1.close()
                r2.close()
            w.close()

def tick_convert(p,tf,start=201812):
    1+1

def parseLineTDM(r):
    l = r.readline()
    if not l: return -1,-1,-1,-1,-1,-1,-1
    s = l.split(',')
    dt= s[0].split(' ')
    d = dt[0].split('.')
    t = dt[1].split(':')
    return int(d[2]),int(d[1]),int(d[0]),int(t[0]),int(t[1]),float(t[2]),float(s[1])

def z(n):
    if n>9: return str(n)
    return "0"+str(n)

def generateTDM(p,tf,start=201812):
    w = open("data/tdm_AU2.csv","w")
    firstOpen = True
    firstReach = False
    oldH = -1
    oldM = -1
    lo = 999999
    hi = -999999
    v = 0
    r = open('data/AU.csv', "r")
    r.readline()
    while True:
        y,m,d,h,n,s,b = parseLineTDM(r)
        if y<0: break
        if firstOpen: 
            o = b
            firstOpen = False
        if ((oldH<23 and abs(h-oldH)>1) or (n%tf < oldM%tf) or (tf==1 and oldM!=n)) and firstReach:
            wd=str(y)+z(m)+z(d)+","+z(h)+":"+z(n)+":"+z(s)+","+str(o)+","+str(hi)+","+str(lo)+","+str(c)+","+str(v)
            if (d==2 and h==0 and n==0): print(wd)
            w.write(wd+"\n")
            o = b
            lo = b
            hi = b
            v = 1
            firstReach=False
        c = b
        if b<lo: lo = b
        if b>hi: hi = b
        v += 1
        if (oldM!=n and firstReach==False): firstReach=True
        oldH = h
        oldM = n
    r.close()
    w.close()

def degap(p,tf):
    if not (p in pairs99):
        try:
            os.rename("data/"+p+str(tf)+".csv","data/"+p+str(tf)+test+".degap.csv")
        except:
            print("file " + p +" not found")
        return

    def open_other_pairs(p):
        C = ["USD","JPY","EUR","GBP","CHF","CAD","AUD","NZD"]
        R = []
        PR= []
        for cc in C:
            pre = p[0:3]
            suf = p[3:6]
            if pre==cc or suf==cc: continue
            pf = []
            pa = []
            for pp in pairs99:
                if pp==pre+cc or pp==cc+pre or pp==suf+cc or pp==cc+suf: 
                    pf.append(open("data/"+pp+str(tf)+".csv","r"))
                    pf[-1].readline()
                    pa.append(pp)
            R.append(pf)
            PR.append(pa)
        return R,PR

    def fill_gap(wr,op,pairs,y1,m1,d1,h1,n1,y2,m2,d2,h2,n2, ooo,ooh,ool,ooc,oooo,oooh,oool,oooc):
        def mergeBid(p,p1,p2,v1,v2):
            pre=p[0:3]
            suf=p[3:6]
            pre1=p1[0:3]
            suf1=p1[3:6]
            pre2=p2[0:3]
            suf2=p2[3:6]
            dp = 5
            if suf=="JPY": dp = 3
            if (pre==pre1 and suf==pre2) or (pre==suf2 and suf==suf1): return [round(v1[0]/v2[0],dp), round(v1[1]/v2[1],dp), round(v1[2]/v2[2],dp), round(v1[3]/v2[3],dp), int((v1[4]+v2[4])/2) ]
            elif (pre==pre2 and suf==pre1) or (pre==suf1 and suf==suf2): return [round(v2[0]/v1[0],dp), round(v2[1]/v1[1],dp), round(v2[2]/v1[2],dp), round(v2[3]/v1[3],dp), int((v2[4]+v1[4])/2) ]
            else: return [round(v2[0]*v1[0],dp), round(v2[1]*v1[1],dp), round(v2[2]*v1[2],dp), round(v2[3]*v1[3],dp), int((v2[4]+v1[4])/2) ]
        
        def average_out(p,v0,v1,v2,v3,v4,v5):
            A =[v0,v1,v2,v3,v4,v5]
            i=0
            o=0
            h=0
            l=0
            c=0
            v=0
            dp = 5
            if p[3:6]=="JPY": dp = 3
            for vv in A:
                if vv[0]>0:
                    i+=1
                    o+=vv[0]
                    h+=vv[1]
                    l+=vv[2]
                    c+=vv[3]
                    v+=vv[4]
            if i==0: return -1,-1,-1,-1,-1
            return round(o/i,dp),round(h/i,dp),round(l/i,dp),round(c/i,dp),int(v/i)

        more = [True,True,True,True,True,True]
        A = []
        for i in range(0,6):
            A.append([[],[]])
            for j in range(0,2):
                A[i][j]={}
                while True:
                    d,t,o,h,l,c,a,v = parseLine(op[i][j])
                    if d<0: break
                    y3=int(d/10000)
                    m3=int((d%10000)/100)
                    d3=d%100
                    h3=int(t.split(":")[0])
                    n3=int(t.split(":")[1])
                    if (y3<y1) or (y3==y1 and m3<m1) or (y3==y1 and m3==m1 and d3<d1) or (y3==y1 and m3==m1 and d3==d1 and h3<h1) or (y3==y1 and m3==m1 and d3==d1 and h3==h1 and n3<n1): continue
                    if (y3>y2) or (y3==y2 and m3>m2) or (y3==y2 and m3==m2 and d3>d2) or (y3==y2 and m3==m2 and d3==d2 and h3>h2) or (y3==y2 and m3==m2 and d3==d2 and h3==h2 and n3>n2): break
                    A[i][j][y3,m3,d3,h3,n3]=[o,h,l,c,v]

        B=[]
        if m2<m1: m2+=12
        for i in range(0,6):
            B.append({})
            for yy in range(y1,y2+1):
                for mm in range(m1,m2+1):
                    mm = ((mm-1)%12)+1
                    dd1 = 1
                    dd2 = 32
                    if y1==y2 and m1==m2 and d1==d2:
                        dd1 = d1
                        dd2 = d1+1
                    for dd in range(dd1,dd2):
                        hh1=0
                        hh2=24
                        if y1==y2 and m1==m2 and d1==d2 and h1==h2:
                            hh1=h1
                            hh2=h1+1
                        for hh in range(hh1,hh2):
                            for nn in range(0,60,tf):
                                key = (yy,mm,dd,hh,nn)
                                if (key in A[i][0]) and (key in A[i][1]):
                                    B[i][key] = mergeBid(p, pairs[i][0],pairs[i][1], A[i][0][key],A[i][1][key])
                                else:
                                    B[i][key] = [-1,-1,-1,-1,-1]
        
        DATA = []
        for yy in range(y1,y2+1):
            for mm in range(m1,m2+1):
                mm = ((mm-1)%12)+1
                dd1 = 1
                dd2 = 32
                if y1==y2 and m1==m2 and d1==d2:
                    dd1 = d1
                    dd2 = d1+1
                for dd in range(dd1,dd2):
                    hh1=0
                    hh2=24
                    if y1==y2 and m1==m2 and d1==d2 and h1==h2:
                        hh1=h1
                        hh2=h1+1
                    for hh in range(hh1,hh2):
                        for nn in range(0,60,tf):
                            key = (yy,mm,dd,hh,nn)
                            o,h,l,c,v=average_out(p,B[0][key],B[1][key],B[2][key],B[3][key],B[4][key],B[5][key])
                            if o<0: continue
                            DATA.append([yy,mm,dd,hh,nn,o,h,l,c,v])

        if len(DATA)==0: return
        dp = 5
        if p[3:6]=="JPY": dp = 3
        offO = (DATA[0][5]-ooo+DATA[-1][5]-oooo)/2
        offH = (DATA[0][6]-ooh+DATA[-1][6]-oooh)/2
        offL = (DATA[0][7]-ool+DATA[-1][7]-oool)/2
        offC = (DATA[0][8]-ooc+DATA[-1][8]-oooc)/2
        oo = ooo
        oh = ooh
        ol = ool
        oc = ooc
        DATA[-1][5] = oooo
        DATA[-1][6] = oooh
        DATA[-1][7] = oool
        DATA[-1][8] = oooc
        for i in range(1,len(DATA)-1):
            #if i==len(DATA)-1:break
            dd = DATA[i]
            o = (dd[5]-offO+oc)/2
            c = (dd[8]-offC+DATA[i+1][5])/2
            h = dd[6]-offH
            if h<max(o,c): h = max(o,c)
            l = dd[7]-offL
            if l>min(o,c): l = min(o,c)

            hr = int(dd[3])
            mo = int(dd[1])
            dy = int(dd[2])
            if mo<3 or (mo==3 and dy<15) or mo>11 or (mo==11 and dy>7): 
                hr = (hr+2)%24
                if hr<=1: dy=dy+1
            else: 
                hr=(hr+3)%24
                if hr<=2: dy=dy+1
            wd=str(dd[0])+z(mo)+z(dy)+","+z(hr)+":"+z(dd[4])+":"+z(0)+","+str(round(o,dp))+","+str(round(h,dp))+","+str(round(l,dp))+","+str(round(c,dp))+","+str(int(dd[9]))
            wr.write(wd+"\n")
            print(p+" filling gap: "+wd)
            oc = c
            
    op62,pairs6 = open_other_pairs(p)
    w = open("data/"+p+str(tf)+test+".degap.csv",'w')
    r = open("data/"+p+str(tf)+".csv",'r')
    r.readline()
    d,t,o,h,l,c,a,v = parseLine(r)
    oy = int(d/10000)
    om = int((d%10000)/100)    
    od = d%100
    oh = int(t.split(":")[0])
    on = int(t.split(":")[1])
    oo = o
    ohh= h
    ol = l
    oc = c
    while True:
        d,t,o,h,l,c,a,v = parseLine(r)
        if d<0: break
        cy = int(d/10000)
        cm = int((d%10000)/100)
        cd = d%100
        ch = int(t.split(":")[0])        
        cn = int(t.split(":")[1])
        #if (ch==5 and cn==30): print(str(d)[0:4]+"-"+str(d)[4:6]+"-"+str(d)[6:])
        #if (cm==2 and od<26 and abs(cd-od)>3) or (cm!=2 and od<28 and abs(cd-od)>3) or (oh<23 and abs(oh-ch)>1) or (on<60-tf and abs(cn-on)>tf): 
        if (cm==2 and od<26 and abs(cd-od)>3) or (cm!=2 and od<28 and abs(cd-od)>3) or (oh<22 and abs(oh-ch)>2): 
            fill_gap(w,op62,pairs6, oy,om,od,oh,on, cy,cm,cd,ch,cn, oo,ohh,ol,oc,o,h,l,c)
        #if ch==oh: print(p+" : rejecting repeating hour at "+str(d)+" "+t)
        else     :
            hr = ch
            yr = int(d/10000)
            mo = int((d%10000)/100)
            dy = d%100
            if mo<3 or (mo==3 and dy<15) or mo>11 or (mo==11 and dy>7): 
                hr = (hr+2)%24
                if hr<=1: dy=dy+1
            else: 
                hr=(hr+3)%24
                if hr<=1: dy=dy+1
            
            wd=z(yr)+z(mo)+z(dy)+","+z(hr)+":"+z(cn)+":"+z(0)+","+str(o)+","+str(h)+","+str(l)+","+str(c)+","+str(v)
            #print(p+":"+wd)
            w.write(wd+"\n")
        oy = cy
        om = cm
        od = cd
        oh = ch
        on = cn
        oo = o
        ohh= h
        ol = l
        oc = c
    w.close()

def generateTF(p):
    def parseLineTF(r):
        d,t,o,hi,lo,c,a,v = parseLine(r)
        if d<0: return -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1
        y = int(d/10000)
        m = int((d%10000)/100)
        d = int(d%100)
        tm= t.split(":")
        h = int(tm[0])
        n = int(tm[1])
        s = int(tm[2])
        #print((y,m,d,h,n,s,o,hi,lo,c,v))
        return y,m,d,h,n,s,o,hi,lo,c,v
    w = open("data/"+p+"_10s.mt4.csv","w")
    #r = open('data/'+p+str(tf)+".degap.csv", "r")
    r = open('data/'+p+"_10s.csv", "r")
    r.readline()
    while True:
        y,m,d,h,n,s,o,hi,lo,c,v = parseLineTF(r)
        if y<0: break
        wd=str(y)+"."+z(m)+"."+z(d)+","+z(h)+":"+z(n)+":"+z(s)+","+str(o)+","+str(hi)+","+str(lo)+","+str(c)+","+str(v)
        if (d==2 and h==0 and n==0): print(wd)
        w.write(wd+"\n")
    r.close()
    w.close()

def load_combi13(p):
    r = open("Data/"+p+str(tf)+".combi13",'rb')
    d = pickle.load(r)
    r.close()
    return np.array(d[0]),np.array(d[1]),np.array(d[2]),d[3],d[4]

def load_target(p):
    r = open("Data/"+p+str(tf)+".combi13",'rb')
    d = pickle.load(r)
    r.close()
    x = []
    for i in range(len(d[0])):
        xx = d[0][i][:286]
        xx = np.append(xx,d[0][i][3718])
        x.append(xx)
    print("combi "+p+" loaded")
    return np.array(x),np.array(d[1]),np.array(d[2]),d[3],d[4]

def load_separate(p):
    r = open("Data/"+p+str(tf)+".combi13",'rb')
    d = pickle.load(r)
    r.close()    
    x = [[],[],[],[],[],[],[],[],[],[],[],[],[]]
    for i in range(len(d[0])):
        for j in range(13):
            xx = d[0][i][j*286:(j+1)*286]
            xx = np.append(xx,d[0][i][3718])
            x[j].append(xx)
    return np.array(x),np.array(d[1]),np.array(d[2]),d[3],d[4]

def merge(p):
    def get_next(r):
        l = r.readline()
        if not l: return -1,-1,-1,-1
        s = l.split(',')
        d = int(s[0])
        t = s[1].split(':')
        h = int(t[0])
        m = int(t[1])
        c = int(t[2])
        return d,h,m,c

    r1 = open('data/'+str(p)+str(tf)+'.csv','r')
    r2 = open('data/'+str(p)+str(tf)+'.tmp.csv','r')
    w = open('data/'+str(p)+str(tf)+'.new.csv','w')
    r1.readline()
    r2.readline()
    d1,h1,m1,c1 = get_next(r1)
    d2,h2,m2,c2 = get_next(r2)
    while (true):
        if d1<0 and d2<0: break
        while (d2<0 and d1>0) or (d1<d2) or (d1==d2 and h1<h2) or (d1==d2 and h1==h2 and m1<m2) or (d1==d2 and h1==h2 and m1<m2 and c1<c2): 
            w.write(l1)
            d1,h1,m1,c1 = get_next(r1)
            if d1<0: break
        while (d1<0 and d2>0)  or (d2<d1) or (d2==d1 and h2<h1) or (d2==d1 and h2==h1 and m2<m1) or (d2==d1 and h2==h1 and m2<m1 and c2<c1): 
            w.write(l2)
            d2,h2,m2,c2 = get_next(r2)
            if d2<0: break
        while d1>0 and d2>0 and d1==d2 and h1==h2 and m1==m2 and c1==c2: 
            w.write(l1)
            d1,h1,m1,c1 = get_next(r1)
            d2,h2,m2,c2 = get_next(r2)
            if d1<0 or d2<0: break

    r1.close()
    r2.close()
    w.close()
    os.remove('data/'+str(p)+str(tf)+'.csv')
    os.rename('data/'+str(p)+str(tf)+'.new.csv', 'data/'+str(p)+str(tf)+'.csv')
    os.remove('data/'+str(p)+str(tf)+'.new.csv')

def prepare_ema(p,t,period):
    r = open('data/'+str(p)+str(t)+'.degap.csv','r')
    r.readline('data/'+str(p)+str(t)+'.csv')

def is_absent(p,yi):
    csvfile = open(p+str(tf)+'.results','r', newline='')
    le = list(csv.reader(csvfile))
    csvfile.close()
    last = len(le)
    for i in range(last):
        if(len(le[i])==0): continue
        d = le[i][0].split(':')[4]
        if ("ensemble" in d): continue
        dyi = int(d.split('/')[2].split('_')[1][:-1])/15-1
        if dyi==yi: return False
    return True

def today_datetime():
    y = int(datetime.today().strftime('%Y'))
    m = int(datetime.today().strftime('%m'))-1
    d = int(datetime.today().strftime('%d'))
    h = time.localtime().tm_hour
    n = time.localtime().tm_min
    s = time.localtime().tm_sec
    return y,m,d,h,n,s