import pandas as pd
import random

class chromossomes_fjsp():

  def __init__(self, data_time, machines, check_factive=False):

    assert isinstance(data_time, pd.DataFrame), '"data_time" must be a pd.DataFrame'
    assert "Ope" in  data_time.columns, 'data_time must presente the column "Ope"'
    assert len(set(machines) - set(data_time.columns))==0, 'the corresponding time M0,M1...Mn must be in data_time columns'

    self.data_time = data_time
    self.machines = machines
    self.MS = self.generate_MS()
    self.OS = self.generate_OS()
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