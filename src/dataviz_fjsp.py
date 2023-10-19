
import matplotlib.pyplot as plt

def plot_gantt(df_input):
  df = df_input.copy()

  df.sort_values(by='Resource', ascending=True, inplace=True)

  # Crie uma lista de rótulos de legenda únicos com cores correspondentes
  unique_labels = df['Job'].unique()
  colors = {
      label: plt.cm.tab20(i) for i, label in enumerate(unique_labels)
  }

  # Crie uma figura e eixos
  fig, ax = plt.subplots(figsize=(20, 12))

  # Loop para criar as barras de Gantt
  for idx, row in df.iterrows():
      color = colors.get(row['Job'], 'gray')
      ax.barh(row['Resource'], row['Finish'] - row['Start'], left=row['Start'], color=color, label = row['Ope']) # label=row['Task'])
      ax.text(row['Start'] + (row['Finish'] - row['Start']) / 2, row['Resource'], row['Ope'], ha='center', va='center')

  # Personalize os eixos
  ax.set_xlabel('Unidades de Tempo')
  ax.set_ylabel('Maquina')
  ax.set_title(f'Gráfico de Gantt - makespan:{df["Finish"].max()}')

  # Crie a legenda com rótulos únicos
  handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in unique_labels]
  ax.legend(handles, unique_labels, loc='upper right')
  ax.invert_yaxis()

  # Exiba o gráfico Gantt
  plt.tight_layout()
  plt.show()