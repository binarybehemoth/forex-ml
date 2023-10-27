import numpy as np
import pygad
import pyswarms as ps
from pyswarm import pso
from sys import argv
import indicators as ind
from symbols import *
import multiprocessing as mp

years = 9

end = 20211204
start = end - years*10000
init = 20120101

def normalize():
   def rd(n):
      return str(round(n/1000000,1))
   w = open("data/USA500IDXUSD1.degap.csv",'w')
   r = open('data/USA500IDXUSD1.csv','r')
   r.readline()
   while True: 
      d,t,o,h,l,c,a,v = ind.parseLine(r,1)
      if d==-1: break
      w.write(str(d)+","+t+","+rd(o)+","+rd(h)+","+rd(l)+","+rd(c)+","+rd(a)+","+str(v)+"\n")
   w.close()
   r.close()

def read_data():
   r = open('data/USA500IDXUSD1.degap.csv','r')
   r.readline()
   d=0
   result = []
   while d<init: 
      d,t,o,h,l,c,a,v = ind.parseLine(r,1)
   while True:
      d,t,o,h,l,c,a,v = ind.parseLine(r,1)
      if d==-1: break
      dd = [d,t,o,h,l,c]
      result.append(dd)
   return result

data= read_data()

def sniper(x):
   global data
   risk = x[0]
   entry = x[1]
   tp = x[2]
   sl = entry * x[3]

   last = len(data)
   i=0
   h=0
   b=1000
   e=1000
   l=0
   ol=0
   t=[]
   j = 0
   to = 0
   #print(x)
   while True:
      if data[i][0]==start: break
      co = data[int(i)][3]
      if (co>h): h=co
      i=i+1

   while i<last-1:
      i=int(i)
      profit=0
      co = data[int(i)][2]
      if (co>h): h=co

      if (b==e):   
         if (co/h<entry):
            l = round(risk*e/co*1000,1)
            profit = -(l-ol)*0.5
      else: 
         if (e/b>tp):
            l=0
            b=e
         elif (co/h<sl):
            l=0
            b=e
            while (i<last-2-15 and data[i+15][2]/h<entry): i+=1
      r = data[i+1][2] - co
      profit = profit + l * r 
      e = e + profit
      if b<30 or e<30: break
      ol = l
      i=i+1
      #if i%1==0:print(str(data[i][0])+" : "+str(data[i][2]/h)+" : "+str(l)+" : "+str(round(e)))
   print(str(round(e)))
   e = e
   if e>30 : g = round(pow(abs(e)/1000,1/years),3)*100
   else: g=0
   return -g


min_cost = 1
best_pos = []
optimizer = 0
maxim = -1

def obj_function_pool(x):
   global min_cost
   global best_pos
   r = pool.map(sniper,x)   
   for j in range(len(r)): 
      if r[j]<min_cost:
         min_cost = r[j]
         best_pos = x[j]
   print(best_pos)
   return r

if __name__ == '__main__':
   pool = mp.Pool(33)
   qq = mp.Queue()
   if argv[1]=="normalize":
      normalize()
   elif argv[1]=="test":
      print(str(len(data))+" records read")
   elif argv[1]=="pso":
      lb = np.array([0.0001, 0.97, 1.001, 0.10])
      ub = np.array([0.02 , 1.0, 1.1, 0.99])
      bounds = (lb, ub)
      options = {'c1': 1.5, 'c2': 3, 'w':0.1}
      optimizer = ps.single.GlobalBestPSO(n_particles=128, dimensions=4, options=options, bounds=bounds)
      cost, pos = optimizer.optimize(obj_function_pool, 10000)
      print(optimizer.swarm.best_pos[optimizer.swarm.pbest_cost.argmin(axis=0)])
      #xopt, fopt = pso(gridder2, lb, ub, maxiter=10, swarmsize=10, debug=True)
      print(pos)
      #print(xopt)
