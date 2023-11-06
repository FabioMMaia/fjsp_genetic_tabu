import pandas as pd
import numpy as np
import math
import random
import tqdm
import sys
sys.path.insert(1, r".")
import matplotlib.pyplot as plt
import pipeline_fjsp

class Chromossome():

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
        
        """
        Initialize the GeneticAlgorithm instance.

        Args:
        data_time (pd.DataFrame): Data or time-related information.
        machines (list): List of available machines.
        Popsize (int): Population size (default: 400).
        maxGen (int): Total generations (default: 200).
        maxStagnantStep (int): Max step size with no improvement (default: 20).
        pr (float): Reproduction probability (default: 0.005).
        pc (float): Crossover probability (default: 0.8).
        pm (float): Mutation probability (default: 0.1).
        maxTSIterSize (int): Max Tabu Search iterations (default: 800).
        maxT (int): Tabu list length (default: 9).
        selection_mode (str): Genetic operator selection mode (e.g., 'elitist_selection').
        """
    
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
            chromossome = Chromossome(self.data_time, self.machines)
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
        aux.sort_values(by='scores', ascending=True, inplace=True)
        selection = aux.head(max(math.ceil(self.Popsize*self.pr),2))
        return selection['pop'].tolist()
    
    def OS_cross_over_POX(self, parent1, parent2, subset=None):

        distinct_values = list(set(parent1))
        if subset is None:
            subset = random.sample(distinct_values, random.randint(1,len(distinct_values)))
        len_OS = len(parent1)

        # Creating child 1
        child1 = [-1] * len_OS
        for value in subset:
            indexes = [i for i, x in enumerate(parent1) if x == value]
            for index in indexes:
                child1[index] = value

        index2 = 0
        for i in range(len_OS):
            if child1[i] == -1:
                while parent2[index2] in subset:
                    index2 += 1
                child1[i] = parent2[index2]
                index2 += 1

        
        # Creating child 2
        child2 = [-1] * len_OS
        for value in subset:
            indexes = [i for i, x in enumerate(parent2) if x == value]
            for index in indexes:
                child2[index] = value

        index2 = 0
        for i in range(len_OS):
            if child2[i] == -1:
                while parent1[index2] in subset:
                    index2 += 1
                child2[i] = parent1[index2]
                index2 += 1

        return [child1, child2]
    
    def OS_cross_over_JBX(self, parent1, parent2, subset=None):

        distinct_values = list(set(parent1))
        if subset is None:
            subset = random.sample(distinct_values, random.randint(1,len(distinct_values)))
        len_OS = len(parent1)

        # Creating child 1
        child1 = [-1] * len_OS
        for value in subset:
            indexes = [i for i, x in enumerate(parent1) if x == value]
            for index in indexes:
                child1[index] = value

        index2 = 0
        for i in range(len_OS):
            if child1[i] == -1:
                while parent2[index2] in subset:
                    index2 += 1
                child1[i] = parent2[index2]
                index2 += 1

        
        # Creating child 2
        child2 = [-1] * len_OS
        subset = list(set(distinct_values) - set(subset))
        for value in subset:
            indexes = [i for i, x in enumerate(parent2) if x == value]
            for index in indexes:
                child2[index] = value

        index2 = 0
        for i in range(len_OS):
            if child2[i] == -1:
                while parent1[index2] in subset:
                    index2 += 1
                child2[i] = parent1[index2]
                index2 += 1

        return [child1, child2]


    def MS_crossover(self,parent1, parent2):
        
        # Select two distinct random crossover points
        point1, point2 = random.sample(range(len(parent1)), 2)
        point1, point2 = min(point1, point2), max(point1, point2)

        # Create children by combining the fragments from parents
        child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
        child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]

        return child1, child2
    
    def OS_mutation_swapping(self,chromosome):
        # Make a copy of the original chromosome for mutation
        mutated_chromosome = chromosome[:]
        len_OS = len(chromosome)
        
        # Step 1: Select two random positions in the chromosome
        position1 = random.randint(0, len(chromosome) - 1)
        position2 = random.randint(0, len(chromosome) - 1)

        while position2==position1:
            position2 = random.randint(0, len(chromosome) - 1)
        
        # Step 2: Swap the elements at the selected positions
        mutated_chromosome[position1], mutated_chromosome[position2] = mutated_chromosome[position2], mutated_chromosome[position1]
        
        return mutated_chromosome
    
    def OS_mutation_neighborhood(self,chromosome):
        # Make a copy of the original chromosome for mutation
        input_chromosome = chromosome[:]
        
        # Randomly select 3 positions from the chromosome
        selected_positions = random.sample(range(len(input_chromosome)), 3)
        
        # Create a list of non-selected positions
        non_selected_positions = [i for i in range(len(input_chromosome)) if i not in selected_positions]
        
        # Shuffle the non-selected positions
        random.shuffle(non_selected_positions)
        
        # Rebuild the original chromosome with the shuffled positions
        permuted_chromosome = [input_chromosome[i] if i in selected_positions else input_chromosome[non_selected_positions.pop(0)] for i in range(len(input_chromosome))]
        
        return permuted_chromosome
        
    def plot_score(self):
        # pd.DataFrame(self.scores_values).boxplot();
        pd.DataFrame(self.scores_values).boxplot(flierprops={'marker': '^', 'markerfacecolor': 'gray', 'markersize': 2})
        plt.xticks([])
        plt.show()

    def MS_mutation(self, MS):

        pos = np.arange(0,len(MS) )
        n_half = round(len(pos)/2)
        selected_pos = random.sample(range(len(pos)), n_half)
        selected_pos.sort()
        mutated_chromosome = MS.copy()

        for pos in selected_pos:
            current_machine = 'M' + str(MS[pos])
            possible_machines = self.data_time.iloc[pos][self.machines].dropna().index.tolist()
            possible_machines.remove(current_machine)
            updated_machine = random.sample(possible_machines, 1)
            updated_machine = int(updated_machine[0][1:])
            mutated_chromosome[pos] = updated_machine

        return mutated_chromosome

    def pipeline_generation(self, improvement_threshold = 1):

        scores_values = {}
        Gen = 0
        best_score_gl=np.inf

        self.init_pop()
        stagnant_count = 0
        scores_values[Gen] = self.scores

        while Gen<self.maxGen and stagnant_count<self.maxStagnantStep:

            # Selection
            selected_genes = self.selection()
            new_pop = []
            new_pop += selected_genes

            while len(new_pop)< self.Popsize:
                parents = random.sample(new_pop, 2)

                # Perform cross over in OS
                if random.random() < self.pc:
                    if random.random() <= 0.5:
                        child1_OS, child2_OS = self.OS_cross_over_POX(parents[0].OS, parents[1].OS)
                    else:
                        child1_OS, child2_OS = self.OS_cross_over_JBX(parents[0].OS, parents[1].OS)
                else:
                    child1_OS = parents[0].OS
                    child2_OS = parents[1].OS

                # Perform cross over in MS
                if random.random() < self.pc:
                    child1_MS,child2_MS  = self.MS_crossover(parents[0].MS, parents[1].MS)
                else:
                    child1_MS,child2_MS = parents[0].MS, parents[1].MS

                    
                child1 = Chromossome(self.data_time, 
                                            self.machines,
                                            MS = child1_MS,
                                            OS = child1_OS)

                child2 = Chromossome(self.data_time, 
                                    self.machines,
                                    MS = child2_MS,
                                    OS = child2_OS)
                    
                # Perform mutation
                if random.random() < self.pm:
                    if random.random() <=0.5:
                        child1.OS = self.OS_mutation_swapping(child1.OS)
                    else:
                        child1.OS = self.OS_mutation_neighborhood(child1.OS)
                if random.random() < self.pm:
                    if random.random()<=0.5:
                        child2.OS = self.OS_mutation_swapping(child2.OS)
                    else:
                        child2.OS = self.OS_mutation_neighborhood(child2.OS)
                if random.random() < self.pm:
                    child1.MS = self.MS_mutation(child1.MS)
                if random.random() < self.pm:
                    child2.MS = self.MS_mutation(child2.MS)
                
                new_pop += [child1, child2]
                
            self.pop = new_pop
            self.update_scores()
            best_score = self.scores.min()

            if best_score < best_score_gl:
                best_score_gl  = best_score
                best_solution = new_pop[self.scores.argmin()]
                stagnant_count=0
            else:
                stagnant_count+=1

            Gen+=1
            scores_values[Gen] = self.scores

            self.scores_values = scores_values

            if Gen%20==0:
                self.plot_score()

        self.best_chromossome = best_solution

        return new_pop
