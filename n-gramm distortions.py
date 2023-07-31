## Lade die benötigten Packages
import os
import pandas as pd
import numpy as np
import re
​
alte_ergebnisse = pd.read_json(r"C:\Users\clalk\JLUbox\Transkriptanalysen\3 N-GRAMME\Analysen\kognitive_verzerrungen.json")
alte_ergebnisse['Class'] = alte_ergebnisse.index
alte_ergebnisse = alte_ergebnisse.reset_index(drop=True)
​
# Verschieben der letzten Spalte nach vorne als erste Spalte
last_column = alte_ergebnisse.pop(alte_ergebnisse.columns[-1])
alte_ergebnisse.insert(0, last_column.name, last_column)
​
# Entfernen der letzten Spalten
num_columns_to_remove = 8
alte_ergebnisse = alte_ergebnisse.drop(alte_ergebnisse.columns[-num_columns_to_remove:], axis=1)
​
​
## lese die vollständigen Patientenabsätze ein
df_pat = pd.read_json('Patientenabsätze_fertig_5.json',orient='split')
​
## durch anzahl der speechturns teilen
for i in range (0, len(alte_ergebnisse)):
    print(i)
    divisor = df_pat['Class'].value_counts()[alte_ergebnisse['Class'][i]]
    alte_ergebnisse.iloc[i, 1:] = alte_ergebnisse.iloc[i, 1:].div(divisor)
​
​
​
alte_ergebnisse.to_json(r'C:\Users\clalk\JLUbox\Transkriptanalysen\3 N-GRAMME\Analysen\kognitive_Verzerrungen_relativ.json')