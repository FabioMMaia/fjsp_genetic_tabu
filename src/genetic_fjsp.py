import pandas as pd
import numpy as np
import math
import random
import tqdm
import sys
sys.path.insert(1, r".")
import pipeline_fjsp

class chromossomes_fjsp():

  def __init__(self, data_time, machines, MS=None, OS=None, check_factive=False):

    assert isinstance(data_time, pd.DataFrame), '"data_time" must be a pd.DataFrame'
    assert "Ope" in  data_time.columns, 'data_time must presente the column "Ope"'
    assert len(set(machines) - set(data_time.columns))==0, 'the corresponding time M0,M1...Mn must be in data_time columns'

    self.data_time = data_time
    self.machines = machines

    if MS is None or OS is None:
        self.MS = self.generate_MS()
        self.OS = self.generate_OS()
    else:
       self.MS = MS
       self.OS = OS
    
    if check_factive:
      self.factivel = self.check_factivel()

  def generate_MS(self):
    MS = []

    for i, row in self.data_time.iterrows():
      possible_machine = row[self.machines].dropna().index.values
      possible_machine = [int(m[1]) for m in possible_machine]
      MS.append(random.choice(possible_machine))
    return MS

  def check_factivel(self):

    factivel = True

    for i, row in self.data_time.iterrows():
      possible_machine = row[self.machines].dropna().index.values
      possible_machine = [int(m[1]) for m in possible_machine]
      if self.MS[i] not in possible_machine:
        return False
    return factivel

  def generate_OS(self):
     operations_vector = self.data_time['Ope'].apply(lambda x: int(x.split('_')[0][1:])).tolist()
     random.shuffle(operations_vector)
     return operations_vector
  
  def show_output(self):
    pipeline_fjsp.run(self.data_time,
                      self.OS,
                      self.MS,
                      verbose=False,
                      plot=True);


class Population():
    def __init__(self, 
                data_time, 
                machines, 
                Popsize=400,
                maxGen=200, 
                maxStagnantStep=20, 
                pr=.005, 
                pc=.8, 
                pm=.1, 
                maxTSIterSize=800, 
                maxT=9, 
                selection_mode = 'elitist_selection'):
    
        self.machines = machines
        self.data_time = data_time
        self.Popsize = Popsize
        self.maxGen = maxGen
        self.maxStagnantStep = maxStagnantStep
        self.pr = pr
        self.pc = pc
        self.pm = pm
        self.maxTSIterSize = maxTSIterSize
        self.maxT = maxT

    def init_pop(self, print_avg=False):

        pop = []
        scores=[]

        for c in tqdm.tqdm(range(self.Popsize)):
            chromossome = chromossomes_fjsp(self.data_time, self.machines)
            pop.append(chromossome)
            _, score = pipeline_fjsp.run(self.data_time, chromossome.OS, chromossome.MS)
            scores.append(score)

        self.pop = pop
        scores = np.array(scores)
        self.scores= scores

        if print_avg:
            print(f'Average score:{round(scores.mean(),2)}')

    def update_scores(self, print_avg=False):
        scores =[]
        for c in tqdm.tqdm(range(self.Popsize)):
            _, score = pipeline_fjsp.run(self.data_time, self.pop[c].OS, self.pop[c].MS)
            scores.append(score)
        scores = np.array(scores)
        self.scores= scores

        if print_avg:
            print(f'Average score:{round(scores.mean(),2)}')

    def selection(self):
        aux = pd.DataFrame({'pop':self.pop, 'scores':self.scores})
        aux.sort_values(by='scores', ascending=True)
        selection = aux.head(max(math.ceil(self.Popsize*self.pr),2))
        return selection['pop'].tolist()
    
    def OS_cross_over_POX(self, parent1, parent2):

        distinct_values = list(set(parent1))
        subset = random.sample(distinct_values, len(distinct_values))

        child = [-1] * len(parent1)
        for value in subset:
            indexes = [i for i, x in enumerate(parent1) if x == value]
            for index in indexes:
                child[index] = value

        index2 = 0
        for i in range(len(parent1)):
            if child[i] == -1:
                while parent2[index2] in subset:
                    index2 += 1
                child[i] = parent2[index2]
                index2 += 1

        return child
    
    def MS_crossover(self,parent1, parent2):
        
        # Select two distinct random crossover points
        point1, point2 = random.sample(range(len(parent1)), 2)
        point1, point2 = min(point1, point2), max(point1, point2)

        # Create children by combining the fragments from parents
        child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
        child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]

        return child1, child2

    def pipeline_generation(self):

        time_series_scores = []
        Gen = 0

        self.init_pop()
        time_series_scores.append(self.scores.mean())

        while Gen<self.maxGen:

            # Selection
            selected_genes = self.selection()
            new_pop = []
            new_pop += selected_genes

            while len(new_pop)< self.Popsize:
                parents = random.sample(new_pop, 2)
                # Reproduction
                child1_OS = self.OS_cross_over_POX(parents[0].OS, parents[1].OS)
                child2_OS = self.OS_cross_over_POX(parents[1].OS, parents[0].OS)
                child1_MS,child2_MS  = self.MS_crossover(parents[0].MS, parents[1].MS)
                # child1_MS = parents[0].MS
                # child2_MS = parents[1].MS

                child1 = chromossomes_fjsp(self.data_time, 
                                            self.machines,
                                            MS = child1_MS,
                                            OS = child1_OS)

                child2 = chromossomes_fjsp(self.data_time, 
                                    self.machines,
                                    MS = child2_MS,
                                    OS = child2_OS)
                
                new_pop += [child1, child2]

            self.pop = new_pop
            self.update_scores()
            time_series_scores.append(self.scores.mean())
            Gen+=1

        self.time_series_scores= time_series_scores
        return new_pop
