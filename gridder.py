import numpy as np
import pygad
import pyswarms as ps
from pyswarm import pso
from sys import argv
import indicators as ind
from symbols import *
import multiprocessing as mp
import csv

years = int(argv[2])
end = 20230707
start = end - years*10000

def convert_csv(infile, outfile):
   def r(s):
      return str(round(float(s),3))
   w = open(outfile,'w')
   with open(infile, newline='') as csvfile:
      reader = csv.reader(csvfile, delimiter=',')
      for row in reader:
         date = row[0].split("-")
         if (len(date)==1): continue
         year = date[0]
         mon  = date[1]
         day  = date[2]
         #print(row)
         w.write(year+mon+day+", 23:59:59, "+r(row[1])+", "+r(row[2])+", "+r(row[3])+", "+r(row[4])+", "+r(row[6])+"\n")
   w.close()

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
   #r = open('data/USA500IDXUSD1.degap.csv','r')
   r = open('sp500_daily.csv','r')
   r.readline()
   d=0
   result = []
   while d<start: 
      d,t,o,h,l,c,a,v = ind.parseLine(r,1)
   while d<end:
      d,t,o,h,l,c,a,v = ind.parseLine(r,1)
      if d==-1: break
      dd = [d,t,o,h,l,c]
      result.append(dd)
   return result

data= read_data()

def gridder(x):
   global data
   distance = x[0]
   wait = int(x[1])
   batch = int(x[2])
   sl = x[3]
   resumeHours = int(x[4])
   skipHours = int(x[5])
   overshoot = x[6]

   last = len(data)
   i=wait
   h=0
   b=1000
   e=1000
   ol=0
   t=[]
   j = 0
   to = 0
   while j<i:
      o = data[int(j)][3]
      if (o>h): 
         h=o
         #print(str(data[j][0])+" : "+str(h))
      j=j+1
   while i<last-1:
      i=int(i)
      wait=int(wait)
      profit=0
      o = data[i][5]
      if (o>h): h=o
      l = round(((h*overshoot)-o)/(distance*o)*b*0.0001/o*2500,1)
      if (data[int(i)][5] < data[int(i-wait*60)][5] and l>ol) or (data[int(i)][5] > data[int(i-wait*60)][5] and l<ol) or (ol-l<0.09999*b/batch and l<ol):
         l=ol
      sum = 0
      for k in range(len(t)):
         sum = sum + t[k][0]
      if (l>ol+0.05):
         t.append([l-ol,o])
         profit=-(l-ol)*0.5
         to = to+1
      elif (l<ol-0.05):
         c=ol-l
         while c>0.005:
            tt=[]
            min=999999
            mi=-1
            for j in range(len(t)):
               if t[j][0] < min:
                  mi=j
                  min=t[j][0]
            if min<c+0.05:
               b = b + (min*(o-t[mi][1]))  - min * 0.5
               if b<0: break
               for j in range(len(t)):
                  if j!=mi: tt.append(t[j])
               t=tt
               c=c-min
            else:
               try:
                  if b<0: break
                  b = b + (c*(o-t[mi][1]))  - c * 0.5 
                  for j in range(len(t)):
                     if j!=mi: tt.append(t[j])
                     else: tt.append([min-c,t[j][1]])
                  t=tt
                  break               
               except:
                  print(str(h*overshoot)+" : "+str(b))
      r = data[i+1][5] - data[i][5]
      profit = profit + l * r 
      e = e + profit
      if b<0 or e<0: break
      #if (i%1==0): print(str(data[i][0])+" : "+data[i][1]+" : "+str(round(b,2))+" : "+str(round(e,2))+" : "+str(o)+" : "+str(h)+" : "+str(l)+" : "+str(ol)+" : "+str(round(profit,2)))

      if 1-(data[i][5]/h)>sl and i>resumeHours*60 and data[i][5] < data[int(i-resumeHours*60)][5]:
         #if argv[1]=="test": print(str(data[i][0])+" : "+str(e))
         for j in range(len(t)):
            profit = t[j][0]*(data[i+1][5]-t[j][1]) - t[j][0] * 0.5
            b = b + profit
            #if argv[1]=="test": print(str(data[i][0])+" : "+str(e)+" : "+str(b)+" : "+str(profit)+" : "+str(data[i+1][5]))
         if argv[1]=="test": print(str(data[i][0])+" : "+str(e)+" : "+str(b))
         l=0
         ol=0
         t=[]
         #print(str(data[i][0])+" : "+data[i][1]+" : "+str(round(b,2))+" : "+str(round(e,2))+" : "+str(o)+" : "+str(data[i][5])+" : "+str(l)+" : "+str(ol)+" : "+str(round(profit,2)))         
         #while 1-(data[i][5]/h)>sl*resume and i<last-3: i+=1
         while data[i][5] < data[int(i-(skipHours+resumeHours)*60)][5] and i<last-3: 
            i+=1
            if (skipHours+resumeHours)*60>i: break
         #i=int(i+skipDays/7*5*22.75*60)
      ol = l
      #print(str(data[int(i)][0])+" : "+data[int(i)][1]+" : "+str(round(b,2))+" : "+str(round(e,2))+" : "+str(l))
      i=i+1
   print(str(round(e/1000,1)))
   e = e
   if e>30 : g = round(pow(abs(e)/1000,1/years),3)*100
   else: g=0
   return -g


def gridderD(x):
   global data
   distance = x[0]
   batch = int(x[1])
   sl = x[2]
   overshoot = x[3]
   rest = int(x[4])
   #   distance           batch          sl               overshoot         rest
   #[6.87330186e-05, 8.35474578e+03, 9.11721122e-01, 1.08630424e+00, 1.33569038e+02]
   last = len(data)
   i=60
   h=0
   b=1000
   e=1000
   ol=0
   t=[]
   j = 0
   to = 0
   while j<i:
      o = data[int(j)][5]
      if (o>h): 
         h=o
         #print(str(data[j][0])+" : "+str(h))
      j=j+1
   while i<last-1:
      i=int(i)
      profit=0
      o = data[i][5]
      if o<1: print(o)
      spread = o/7000
      if (o>h): h=o
      mbe = b
      if e>b: mbe = e
      l = round(((h*overshoot)-o)/(distance*o)*mbe/o*0.25,1)
      if h*overshoot<o: l = 0
      sum = 0
      for k in range(len(t)):
         sum = sum + t[k][0]
      if (l>ol+0.09):             # open BUY trades
         t.append([l-ol,o])
         profit=-(l-ol)*spread
         to = to+1
      elif (l<ol-0.09*b/batch/o*4000):   # close BUY trades
         c=ol-l
         while c>0.005:
            if (len(t)==0): break
            tt=[]
            min=999999
            mi=-1
            for j in range(len(t)):
               if t[j][0] < min:
                  mi=j
                  min=t[j][0]
            if min<c+0.05:
               b = b + (min*(o-t[mi][1]))  - min * spread
               if b<0: break
               for j in range(len(t)):
                  if j!=mi: tt.append(t[j])
               t=tt
               c=c-min
            else:
               try:
                  if b<0: break
                  b = b + (c*(o-t[mi][1]))  - c * spread
                  for j in range(len(t)):
                     if j!=mi: tt.append(t[j])
                     else: tt.append([min-c,t[j][1]])
                  t=tt
                  break               
               except:
                  print(str(h*overshoot)+" : "+str(b)+" : "+str(t)+" : "+str(min)+" : "+str(j))
      if b<10 or e<10: break                  
      if e/b<sl: 
         ol = 0
         profit = 0
         l = 0
         t = []
         b = e
         i = i + rest
         if (i>=len(data)): break
         h = data[i][5]
      else:
         r = data[i+1][5] - data[i][5]
         profit = profit + l * r 
         e = e + profit
         ol = l
         i=i+1
         #print(b)
         #print(e)
         #print(l)
         #print(t)
         #print(h)
         #print(data[i])
         #print(data[i+1])
   #print(str(round(e/1000,1)))
   e = e
   if e>30 : g = round(pow(abs(e)/1000,1/years)-1,3)
   else: g=0
   return -g


def gridderDS(x):
   global data
   distance = x[0]
   batch = int(x[1])
   save = x[2]
   overshoot = x[3]
   rest = int(x[4])
   restart = x[5]

   last = len(data)
   i=60
   h=0
   b=1000
   e=1000
   ol=0
   t=[]
   j = 0
   to = 0
   saving = 0
   while j<i:
      o = data[int(j)][3]
      if (o>h): 
         h=o
         #print(str(data[j][0])+" : "+str(h))
      j=j+1
   while i<last-1:
      if data[i][0] % 10000 == 102:
         s = e*save
         saving = saving + s
         b = b - s
         e = e - s
      i=int(i)
      profit=0
      o = data[i][5]
      spread = o/7000
      if (o>h): h=o
      mbe = b
      if e>b: mbe = e
      l = round(((h*overshoot)-o)/(distance*o)*b/o*0.25,1)
      sum = 0
      for k in range(len(t)):
         sum = sum + t[k][0]
      if (l>ol+0.09):             # open BUY trades
         t.append([l-ol,o])
         profit=-(l-ol)*spread
         to = to+1
      elif (l<ol-0.09*b/batch/o*4000):   # close BUY trades
         c=ol-l
         while c>0.005:
            if (len(t)==0): break
            tt=[]
            min=999999
            mi=-1
            for j in range(len(t)):
               if t[j][0] < min:
                  mi=j
                  min=t[j][0]
            if min<c+0.05:
               b = b + (min*(o-t[mi][1]))  - min * spread
               if b<0: break
               for j in range(len(t)):
                  if j!=mi: tt.append(t[j])
               t=tt
               c=c-min
            else:
               try:
                  if b<0: break
                  b = b + (c*(o-t[mi][1]))  - c * spread
                  for j in range(len(t)):
                     if j!=mi: tt.append(t[j])
                     else: tt.append([min-c,t[j][1]])
                  t=tt
                  break               
               except:
                  print(str(h*overshoot)+" : "+str(b)+" : "+str(t)+" : "+str(min)+" : "+str(j))
      if b<10 or e<10: break                  
      if e<100: 
         ol = 0
         profit = 0
         l = 0
         t = []
         s = saving * restart
         saving = saving - s
         b = b+s
         e = e+s
         i = i + rest
         if (i>=len(data)): break
         h = data[i][5]
         continue
      r = data[i+1][5] - data[i][5]
      profit = profit + l * r 
      e = e + profit
      ol = l
      i=i+1
   #print(str(round(e/1000,1)))
   e = e
   if e>30 : g = round(pow(abs(e+saving)/1000,1/years)-1,3)
   else: g=0
   return -g

min_cost = 1
best_pos = []
optimizer = 0
maxim = -1

def obj_function_pool(x):
   global min_cost
   global best_pos
   r = pool.map(gridderD,x)   
   for j in range(len(r)): 
      if r[j]<min_cost:
         min_cost = r[j]
         best_pos = x[j]
         print("\n"+str(round(best_pos[0],4))+", "+str(round(best_pos[1]))+", "+str(round(best_pos[2],3))+", "+str(round(best_pos[3],3))+", "+str(round(best_pos[4])))
   return r

def obj_function_pool_saving(x):
   global min_cost
   global best_pos
   r = pool.map(gridderDS,x)   
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
      #gridder(0.008723, 102, 4538, 0.616, 34.5, 1.0290)
      #             distance          wait           batch               sl          skipDays         overshoot
      #gridderD([4.09790587e-04, 1.05934336e+01, 7.43316487e+03, 6.11390493e-02, 8.82679211e+02, 6.83689487e+02, 1.06401530e+00])
      #           distance           batch          sl               overshoot         rest
      gridderD([1.40055130e-04, 2.34737856e+03, 3.77202599e-02, 9.93401194e-01, 3.24754426e+02])
   elif argv[1]=="convert_csv":
      convert_csv("sp500_daily_prices.csv", "sp500_daily.csv")
   elif argv[1]=="pso":
      lb = np.array([0.0001,   1, 1000, 0.03, 0.95])
      ub = np.array([0.01  , 120,10000, 0.25, 1.05])
      bounds = (lb, ub)
      options = {'c1': 1.5, 'c2': 3, 'w':0.1}
      optimizer = ps.single.GlobalBestPSO(n_particles=256, dimensions=7, options=options, bounds=bounds)
      cost, pos = optimizer.optimize(obj_function_pool, 10000)
      print(optimizer.swarm.best_pos[optimizer.swarm.pbest_cost.argmin(axis=0)])
      #xopt, fopt = pso(gridder2, lb, ub, maxiter=10, swarmsize=10, debug=True)
      print(pos)
      #print(xopt)
   elif argv[1]=="psoD":
      #             distance  batch sl   overshoot  rest
      lb = np.array([0.0001,    1, 0,     0.9,       1])
      ub = np.array([0.1   ,10000, 1,     1.1,     500])
      bounds = (lb, ub)
      options = {'c1': 5, 'c2': 1, 'w':0.1}
      optimizer = ps.single.GlobalBestPSO(n_particles=256, dimensions=5, options=options, bounds=bounds)
      cost, pos = optimizer.optimize(obj_function_pool, 100000)
      #print(optimizer.swarm.best_pos[optimizer.swarm.pbest_cost.argmin(axis=0)])
      #xopt, fopt = pso(gridder2, lb, ub, maxiter=10, swarmsize=10, debug=True)
      print(pos)
      #print(xopt)
   elif argv[1]=="psoDS":
      # [1.774e-03 1.411e+03 6.963e-01 1.067e+00 3.589e+02]
      #              distane  batch daily saving   overshoot  rest  restart
      lb = np.array([0.00001,    1,       0,           0.9,       1,   0])
      ub = np.array([0.01   ,10000,     0.1,           1.1,     500,   1])
      bounds = (lb, ub)
      options = {'c1': 5, 'c2': 1, 'w':0.1}
      optimizer = ps.single.GlobalBestPSO(n_particles=256, dimensions=6, options=options, bounds=bounds)
      cost, pos = optimizer.optimize(obj_function_pool_saving, 100000)
      #print(optimizer.swarm.best_pos[optimizer.swarm.pbest_cost.argmin(axis=0)])
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