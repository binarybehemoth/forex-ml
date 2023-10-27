import numpy as np
import pygad
import pyswarms as ps
from sys import argv
import indicators as ind
from symbols import *
import multiprocessing as mp
import csv

years = 20
end = 20230707
start = end - years*10000

def read_data():
   #r = open('data/USA500IDXUSD1.degap.csv','r')
   r = open('sp500_daily.csv','r')
   r.readline()
   d=0
   result = []
   while d<start: 
      d,t,o,h,l,c,a,v = ind.parseLine(r,1)
      #print(str(d))
   while d<end:
      d,t,o,h,l,c,a,v = ind.parseLine(r,1)
      #print(str(d)+" "+str(start))
      if d==-1: break
      dd = [d,t,o,h,l,c]
      result.append(dd)
   return result

data= read_data()

growth = []
gi = []

if __name__ == '__main__':
   for m in range(13):
      growth.append([])
      gi.append([])
      for d in range(32):
         growth[m].append(0)
         gi[m].append(0)
   cc = data[0][5]
   print(str(data[0][0])+" "+str(data[0][5]))
   print(str(data[-1][0])+" "+str(data[-1][5]))
   for dat in data:
      c = dat[5]
      #print(str(dat[0])+" "+str(c))
      m = int(dat[0]/100)%100
      d = dat[0]%100
      #if d!=28: continue
      growth[m][d] = growth[m][d] + c/cc-1
      gi[m][d] = gi[m][d]+1
      cc = c
      #print(str(dat[0])+ " "+str(m)+" "+str(d)+"---"+str(gi[m][d]))
   gg = 0
   ggg = 0
   e = 100
   for m in range(1,13):
      for d in range(1,32):
         if gi[m][d]==0: continue
         g = growth[m][d]/gi[m][d]
         e = e+e*g;
         #print(str(m)+":"+str(d)+", "+str(round( (gg+ggg+g)/3*100, 3)))
         print(str(m)+":"+str(d)+", "+str(round(g*100, 3)))
         ggg = gg
         gg = g
   print("CAGR: "+str(e-100));
