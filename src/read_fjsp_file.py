
import pandas as pd

def parse_data(fjsp_data, return_descr=False):
  ope = []
  machines_time = []

  for n_row, row in enumerate(fjsp_data.split('\n')):

    if n_row==0:
      N, A, K = [int(item) for item in row.split()]
      descr = {}
      descr['N'] = N
      descr['A'] = A
      descr['K'] = K

    elif n_row>0 and n_row<A+1:
      ope.append([int(item) for item in row.split()])

    elif len(row)>0:
      machines_time.append([int(item) for item in row.split()])

  df_fjsp_data = pd.DataFrame(machines_time)
  ope_df = pd.DataFrame(ope)
  cols = []

  for i in range(int((df_fjsp_data.shape[1]-1)/2)):
    cols += [f'm_{str(i)}', f't_m_{str(i)}']
  df_fjsp_data.columns = ['tot_m'] + cols

  df_fjsp_data.reset_index(inplace=True)
  df_fjsp_data = df_fjsp_data.merge(ope_df, left_on = 'index', right_on = 0, how='left')
  df_fjsp_data.drop(columns=[0], inplace=True)
  df_fjsp_data.rename(columns={1:'next_ope'}, inplace=True)
  df_fjsp_data = df_fjsp_data.merge(ope_df, left_on = 'index', right_on = 1, how='left')
  df_fjsp_data.drop(columns=[1], inplace=True)
  df_fjsp_data.rename(columns={0:'previous_ope', 'index':'ope'}, inplace=True)
  df_fjsp_data = df_fjsp_data.sort_index(axis=1)

  df_fjsp_data['job'] = df_fjsp_data['previous_ope'].isna().astype(int)
  df_fjsp_data['job'] =  df_fjsp_data['job'].cumsum()

  df_fjsp_data.fillna(-1,inplace=True)
  df_fjsp_data_t = df_fjsp_data.drop(columns=df_fjsp_data.columns[df_fjsp_data.columns.str.startswith('t_m_')].to_list()).melt(id_vars=['ope', 'next_ope', 'previous_ope', 'tot_m', 'job'], var_name='var', value_name='machine')
  df_fjsp_data_t = df_fjsp_data_t.merge(df_fjsp_data.drop(columns=df_fjsp_data.columns[df_fjsp_data.columns.str.startswith('m_')].to_list()).melt(id_vars=['ope', 'next_ope', 'previous_ope', 'tot_m', 'job'], var_name='var', value_name='t_m')['t_m'], right_index=True, left_index=True )
  df_fjsp_data_t['machine'] = 'M' + df_fjsp_data_t['machine'].astype(int).astype(str)

  df_fjsp_data_t= df_fjsp_data_t.pivot_table(index=['ope', 'next_ope', 'previous_ope', 'tot_m', 'job'], columns='machine', values='t_m', aggfunc='first').reset_index(drop=False)
  df_fjsp_data_t['job_ope'] = df_fjsp_data_t.groupby('job').cumcount()+1
  df_fjsp_data_t['index'] = df_fjsp_data_t.apply(lambda row: 'O' + str(int(row['job'])) + '_' + str(int(row['job_ope'])), axis=1)
  df_fjsp_data_t['job']  = 'Job-' + df_fjsp_data_t['job'].astype(str)
  df_fjsp_data_t.drop(columns=['M-1'], inplace=True)
  df_fjsp_data_t.rename(columns={'index':'Ope', 'job':'Job'},inplace=True)
  df_fjsp_data_t.set_index(['Job','Ope'] , inplace=True)
  df_fjsp_data_t = df_fjsp_data_t[df_fjsp_data_t.columns[df_fjsp_data_t.columns.str.startswith('M')]]
  df_fjsp_data_t.reset_index(inplace=True)

  if return_descr:
    return df_fjsp_data_t, descr
  else:
    return df_fjsp_data_t