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

if __name__ == '__main__':
   for y in range(2024):
      growth.append([])
      for m in range(13):
         growth[y].append([])
         for d in range(32):
            growth[y][m].append(0)
   cc = data[0][5]
   for dat in data:
      c = dat[5]
      #print(str(dat[0])+" "+str(c))
      y = int(dat[0]/10000)
      m = int(dat[0]/100)%100
      d = dat[0]%100
      #if d!=28: continue
      growth[y][m][d] = c/cc-1
      cc = c
      #print(str(dat[0])+ " "+str(m)+" "+str(d)+"---"+str(gi[m][d]))
   gg = 0
   ggg = 0
   e = 100
   
   for m in range(1,13):
      for d in range(1,32):
         s = str(m)+"/"+str(d)+"/"+str(y)+","
         for y in range(2004,2024):       
            g = growth[y][m][d]
            s = s + str(round(g*100,3)) + ","
         print(s)
   print("CAGR: "+str(e-100));
