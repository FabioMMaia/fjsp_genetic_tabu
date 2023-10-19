import sys
sys.path.insert(1, r".")
import dataviz_fjsp
import pandas as pd

def run(data_time, OS, MS, verbose=False, plot=False):

  assert isinstance(data_time, pd.DataFrame), "data_time must be a pd.DataFrame containing the columns Job, Ope, Mi, M_i+1 ... corresponding to the time needed to perfome operation"
  assert isinstance(OS, list) and len(OS) == len(data_time['Ope']), 'OS must be de gene must be a list responsible for ordering having the same length as the number of operation'
  assert isinstance(MS, list) and len(OS) == len(data_time['Ope']), 'MS must be de gene must be a list responsible for machine assigment having the same length as the number of operation'

  # df = pd.DataFrame(data_time)
  df = data_time.copy()
  df.set_index(['Ope'], inplace=True)
  machines = {}
  M = 0
  new_start = 0

  for machine in df.columns[df.columns.str.startswith('M')]:
    m = Machine(machine, df[machine])
    machines[machine] = m
    M+=1

  df = pd.DataFrame(data_time)

  #Inicializa dataframe
  unique_jobs = df['Job'].unique().tolist()
  tasks = {i + 1: job for i, job in enumerate(unique_jobs)}
  # all_ope = df['Ope'].sort_values().tolist()
  all_ope = df['Ope'].tolist() # certificar se Oij esta na ordem correta
  executed_Op = []
  all_machines = df.columns[df.columns.str.startswith('M')].tolist()
  unavailable_times_machines = {key: [] for key in all_machines}
  output = pd.DataFrame()

  for i in range(0, len(OS)):
    row = {}
    row['Job'] = tasks[OS[i]]
    Oi = str(OS[i])
    Oj = str(len([x for x in OS[:i+1] if x in [OS[i]]]))
    Op = 'O' + Oi + '_' + Oj
    row['Ope'] = Op
    executed_Op.append(Op)

    if Op.split('_')[-1]!='1':
      previous_Op = Op.split('_')[0] + '_' + str(int(Op.split('_')[-1])-1)
    else:
      previous_Op=None

    row['previous_Ope'] = previous_Op
    machine =  'M' + str(MS[all_ope.index(Op)])
    row['Resource'] = machine
    time_ope_M = df.query('Ope==@Op')[machine].iloc[0]


    if time_ope_M != time_ope_M:
      print('Solução Não Factível')
      return None

    row['Time_Execution'] = time_ope_M

    if previous_Op is None:
      machines[machine].assign_operation(Op)

    else:
      for m in machines.keys():
        for Op_ in machines[m].operations.keys():
          if previous_Op == Op_:
            new_start = machines[m].operations[previous_Op]['End']
            machines[machine].assign_operation(Op, new_start)
            break

    row['Start'] =  machines[machine].operations[Op]['Start']
    row['Finish'] =  machines[machine].operations[Op]['End']

    output = pd.concat([output,pd.DataFrame(row, index=[0])], axis=0, ignore_index=True)
    if verbose:
      print(row)
      print(machines)
      print('------------#--------------')

  if plot:
    dataviz_fjsp.plot_gantt(output)

  makespan = output['Finish'].max()

  return output, makespan


class Machine():
  def __init__(self, machine_name, time_execution):
    self.machine_name = machine_name
    self.time_execution = time_execution
    self.list_unavailability = []
    self.operations = {}

  def assign_operation(self, operation, start=None):
    time_execution_op = self.time_execution.loc[operation]

    # If there is no constrain about when start operation (no pre-requirement)
    if start is None:
      t = 0
    else:
      t = start
    # Check if is the first operation of the machine
    if len(self.operations.keys())==0:
      self.operations[operation] = {'Time exec.': time_execution_op, 'Start': t, 'End': t +  time_execution_op}
      self.list_unavailability.append([t,t +time_execution_op])

    # If the machine has already performed an operation
    else:
      ta, tb = self.find_best_interval(t,time_execution_op, self.list_unavailability)
      self.operations[operation] = {'Time exec.': time_execution_op, 'Start': ta, 'End': ta +  time_execution_op}
      self.list_unavailability.append([ta, ta +  time_execution_op])


  def find_best_interval(self, tmin, delta_t, intervalos):
    intervalos_ordenados = sorted(intervalos, key=lambda x: x[0])

    tmin_final = tmin
    encontrado = False

    for intervalo in intervalos_ordenados:
        t1, t2 = intervalo
        if tmin_final + delta_t <= t1:
            encontrado = True
            break
        tmin_final = max(tmin_final, t2)

    if not encontrado:
        if not intervalos_ordenados:
            tmin_final = tmin
        else:
            tmin_final = max(tmin_final, intervalos_ordenados[-1][1])

    return tmin_final, tmin_final + delta_t


  def __repr__(self):
    return f"Machine(machine_name='{self.machine_name}',  operations={self.operations}')"