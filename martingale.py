import numpy as np
import pygad
import pyswarms as ps
from pyswarm import pso
from sys import argv
import indicators as ind
from symbols import *
import multiprocessing as mp

years = 10

#end = 20211204
end = 20221013
start = end - years*10000

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
   while d<start: 
      d,t,o,h,l,c,a,v = ind.parseLine(r,1)
   while True:
      d,t,o,h,l,c,a,v = ind.parseLine(r,1)
      if d==-1: break
      dd = [d,t,o,h,l,c]
      result.append(dd)
   return result

data= read_data()

def martingale(x):
   def buy_empty():
      nonlocal t
      for tt in t:
         if tt['typ']=="B": return False
      return True

   def sell_empty():
      nonlocal t
      for tt in t:
         if tt['typ']=="S": return False
      return True

   def buy_first():
      nonlocal t
      t.append({'typ':"B", 'price':o+spread/2, 'lot':max(0.1,round(b*risk,1))})

   def sell_first():
      nonlocal t
      t.append({'typ':"S", 'price':o-spread/2, 'lot':max(0.1,round(b*risk,1))})

   def close_all_buy():
      nonlocal t,b
      ttt=[]
      for tt in t:
         if tt['typ']=="S": ttt.append(tt)
      t=ttt
      b=b+tbp

   def close_all_sell():
      nonlocal t,b
      ttt=[]
      for tt in t:
         if tt['typ']=="B": ttt.append(tt)
      t=ttt
      b=b+tsp

   def lowest_buy():
      nonlocal t
      min=999999
      v=0
      for tt in t:
         if tt['typ']=="B" and tt['price']<min: 
            min = tt['price']
            v = tt['lot']
      return [min,v]

   def highest_sell():
      nonlocal t
      max=-999999
      v=0
      for tt in t:
         if tt['typ']=="S" and tt['price']>max: 
            max = tt['price']
            v = tt['lot']
      return [max,v]

   def buy_next(v):
      nonlocal t
      t.append({'typ':"B", 'price':o+spread/2, 'lot':max(0.1,round(v*exp,1))})

   def sell_next(v):
      nonlocal t
      t.append({'typ':"S", 'price':o-spread/2, 'lot':max(0.1,round(v*exp,1))})

   def total_buy_profits():
      nonlocal t,o
      p = 0
      for tt in t:
         if tt['typ']=="B": p = p + tt['lot'] * ((o-spread/2)-tt['price'])
      return p

   def total_sell_profits():
      nonlocal t,o
      p = 0
      for tt in t:
         if tt['typ']=="S": p = p + tt['lot'] * (tt['price']-(o+spread/2))
      return p

   global data
   risk = x[0]
   distance = x[1]
   exp = x[2]
   tp = x[3]
   sl = x[4]
   wait = int(x[5])
   surge = x[6]

   last = len(data)
   i=0
   b=1000    # balance
   e=1000    # equity
   eMax=e    # equity max
   minEP=1   # minimum equity proportion
   ol=0      # old lots
   t=[]      # trades

   j=0
   ow=[]

   while i<wait: 
      i=int(i)
      o = data[i][2]      
      ow.append(o)
      i=i+1

   while i<last-1:
      i=int(i)
      o = data[i][2]
      w = ow[0]
      ow = ow[1:]
      ow.append(o)
      spread = max(round(o/6000,1),0.2)

      if e<100: 
         print("bust at "+str(data[i][0]))
         return 999999
      if e/b<sl:
         t = []
         b = e
      if e      > eMax : eMax = e
      if e/eMax < minEP: minEP = e / eMax

      isRising = (o>w)
      tbp = total_buy_profits()
      tsp = total_sell_profits()
      if buy_empty()  and      isRising : buy_first()
      if sell_empty() and (not isRising): sell_first()
      if      isRising and o>ow[-2]+surge*o: close_all_sell()
      if (not isRising) and o<ow[-2]-surge*o: close_all_buy()
      [lb, v] = lowest_buy()
      if isRising and o<lb - distance*o: buy_next(v)
      elif tbp>tp*b: close_all_buy()
      [hs, v] = highest_sell()
      if (not isRising) and o>hs + distance*o: sell_next(v)
      elif tsp>tp*b: close_all_sell()
      e = b + tbp + tsp
      i=i+1
      #print(data[i][0],data[i][1],o,round(b,2),round(e,2),round(tp,2),tp*b,round(tbp,2),round(tsp,2),t)
      #if i%10000==0: print(data[i][0],o,round(e,2),t)
   print(round(e))
   return -e


min_cost = 1
best_pos = []
optimizer = 0
maxim = -1

def obj_function_pool(x):
   global min_cost
   global best_pos
   r = pool.map(martingale,x)   
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
      martingale([2.55694721e-04, 5.91813579e-02, 2.89255651e+00, 1.59839499e-02, 7.42409324e-02, 1.06277872e+02, 2.50461312e-02])
      martingale([1.61691665e-04, 3.11414662e-02, 4.26896235e+00, 6.65749179e-02, 5.91987482e-01, 2.74519767e+01, 2.53710627e-02])
   elif argv[1]=="pso":
                     # risk  distance  exp    tp    sl    wait  surge
      lb = np.array([0.0001,  0.001,  1.05, 0.001, 0.05,    2, 0.005])
      ub = np.array([0.001 ,    0.1,     5,   0.1,  0.9 , 1440, 0.03 ])
      bounds = (lb, ub)
      options = {'c1': 1.5, 'c2': 3, 'w':0.1}
      optimizer = ps.single.GlobalBestPSO(n_particles=256, dimensions=7, options=options, bounds=bounds)
      cost, pos = optimizer.optimize(obj_function_pool, 10000)
      print(optimizer.swarm.best_pos[optimizer.swarm.pbest_cost.argmin(axis=0)])
      #xopt, fopt = pso(gridder2, lb, ub, maxiter=10, swarmsize=10, debug=True)
      print(pos)
      #print(xopt)

   elif argv[1]=="ga":
      lb = np.array([0.0001,   1, 1000, 0.1,  1, 1.01])
      ub = np.array([0.01  , 120,10000, 0.8, 60, 1.2])      
      def fitness_func(f,i):
         return gridder(i,f[0],f[1],f[2],f[3],f[4],f[5])
      gene_space = [{'low':0.0001, 'high':0.01, 'step':0.0001},
                     {'low':1, 'high':120, 'step':1},
                     {'low':1000, 'high':10000, 'step':100},
                     {'low':0.1, 'high':0.8, 'step':0.01},
                     {'low':1, 'high':60, 'step':1},
                     {'low':1.01, 'high':1.2, 'step':0.01}]         
      fitness_function = fitness_func
      num_generations = 10
      num_parents_mating = 4
      sol_per_pop = 8
      num_genes = 6
      parent_selection_type = "sss"
      keep_parents = 1
      crossover_type = "single_point"
      mutation_type = "random"
      mutation_percent_genes = 30
      ga_instance = pygad.GA(
                       gene_space=gene_space,
                       num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes)

      ga_instance.run()

      solution, solution_fitness, solution_idx = ga_instance.best_solution()
      print("Parameters of the best solution : {solution}".format(solution=solution))
      print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

      prediction = gridder(0,solution[0],solution[1],solution[2],solution[3],solution[4],solution[5])
      print("Predicted output based on the best solution : {prediction}".format(prediction=prediction))