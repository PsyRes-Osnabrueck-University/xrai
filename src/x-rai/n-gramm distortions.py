## Lade die benötigten Packages
import os
import pandas as pd
import numpy as np
import re
import json




# Patientenabsätze erstellen
os.chdir(r"C:\Users\clalk\JLUbox\Transkriptanalysen\Alle Transkripte\Transkripttabellen")
df_full = pd.read_json('Transkripttabelle vollständig.json',orient='records')
df_pat = df_full.loc[:, ["File", "Therapist_no", "Patient_no", "Session", "Patient"]]
df_pat = df_pat.explode("Patient", ignore_index=True)
df_pat['File'] = df_pat['File'].apply(lambda x: x.rstrip(".txt") if x.endswith(".txt") else x)
df_pat = df_pat.loc[:, ["File", "Session", "Patient"]]
df_pat.columns=["Class", "session", "Patient"]
df_pat.to_json("Patientenabsätze vollständig.json")
df_pat.to_excel("Patientenabsätze vollständig.xlsx")


os.chdir(r"C:\Users\clalk\JLUbox\Transkriptanalysen\Alle Transkripte\Transkripttabellen")

df_pat = pd.read_excel("Patientenabsätze vollständig.xlsx")
df_pat = pd.read_excel("Patientenabsätze_5.xlsx")

## Kleinschreibung für df_pat
df_pat["Patient"] = df_pat["Patient"].str.lower()

## lese das Wörterbuch ein
kognitive_verzerrungen = pd.read_excel(r"C:\Users\clalk\JLUbox\Transkriptanalysen\3 KOGNITIVE VERZERRUNGEN\Wörterbuch_deutsch_fin.xlsx")
kognitive_verzerrungen = kognitive_verzerrungen.rename(columns={'Unnamed: 2': 'Verzerrung 2', 'Unnamed: 3': 'Verzerrung 3','Unnamed: 4': 'Verzerrung 4', 'Unnamed: 5': 'Verzerrung 5', 'Unnamed: 6': 'Verzerrung 6', 'Unnamed: 7': 'Verzerrung 7', 'Unnamed: 8': 'Verzerrung 8', 'Unnamed: 9': 'Verzerrung 9', 'Unnamed: 10': 'Verzerrung 10'})
# Kleinschreibung auf kognitive_verzerrungen anwenden
kognitive_verzerrungen = kognitive_verzerrungen.applymap(lambda s:s.lower() if type(s) == str else s)

# Kognitive Verzerrungen: in ein Dataframe mit nur zwei Spalten übertragen, damit Auswertung leichter ist.Schleife durchführen und Werte in den neuen Dataframe übertragen
# Neue Dataframe erstellen
new_kognitive_verzerrungen = pd.DataFrame(columns=['Kategorie', 'Verzerrung'])

# Schleife durchführen und Werte in den neuen Dataframe übertragen
for index, row in kognitive_verzerrungen.iterrows():
    new_rows = []
    for i in range(1, len(row)):
        new_rows.append({'Kategorie': row[0], 'Verzerrung': row[i]})
    new_kognitive_verzerrungen = pd.concat([new_kognitive_verzerrungen, pd.DataFrame(new_rows)], ignore_index=True)

# Drop NAs
new_kognitive_verzerrungen = new_kognitive_verzerrungen.dropna(subset=['Verzerrung'])

# Drop Duplicates (Spalte 2)
new_kognitive_verzerrungen = new_kognitive_verzerrungen.drop_duplicates(subset=['Verzerrung'])

# Den Index wieder normal machen
new_kognitive_verzerrungen = new_kognitive_verzerrungen.reset_index(drop=True)
dict_verzerrung = {}

# Dictionary mit Einträgen für jede Verzerrung erstellen
for kategorie in new_kognitive_verzerrungen["Kategorie"].unique():
    dict_verzerrung[kategorie] = new_kognitive_verzerrungen.loc[new_kognitive_verzerrungen["Kategorie"]==kategorie]["Verzerrung"].tolist()


## Kleinschreibung für df_pat
df_pat["Patient"] = df_pat["Patient"].str.lower()

#df_pat = df_pat.drop("Unnamed: 0", axis=1)


for category in kognitive_verzerrungen["Kategorie"].unique():
    # Überprüfe, ob der Spaltenname schon im DataFrame existiert
    if category not in df_pat.columns:
        # Füge eine neue Spalte hinzu
        df_pat[category] = None
df_pat["I_talk"] = None
df_pat["words"] = None

## fülle new_df_pat mit Nullen
df_pat = df_pat.fillna(0)
#df_pat = df_pat.drop(133638, axis=0)

dict_pat = df_pat.to_dict()


### prüfe, ob die kognitiven Verzerrungen in den Patientenabsätzen vorkommen
# outer loop
for i in range(len(dict_pat["Class"])):
    print(i)
# inner loop
    for kategorie in dict_verzerrung:
        for verzerrung in dict_verzerrung[kategorie]:
            if verzerrung in dict_pat["Patient"][i]:
                dict_pat[kategorie][i] = 1
                break

for i in range(len(dict_pat["Class"])):
    print(i)
# inner loop
    for pronomen in ["ich", "mein", "meine", "meiner", "meines", "meins", "mir", "mich"]:
        if pronomen in dict_pat["Patient"][i]:
            dict_pat["I_talk"][i] =1
            break

for i in range(len(dict_pat["Class"])):
    print(i)
    dict_pat["words"][i] = len(dict_pat["Patient"][i])-1
# inner loop
    for pronomen in ["ich", "mein", "meine", "meiner", "meines", "meins", "mir", "mich"]:
        if pronomen in dict_pat["Patient"][i]:
            dict_pat["I_talk"][i] =1
            break


df_pat = pd.DataFrame.from_dict(dict_pat)
# ----- Distortions

#df_pat_aggregated = df_pat.drop(["session", "Patient"], axis=1)
df_words = df_pat[["Class", "words"]]
df_words = df_words.groupby("Class").agg("sum")
df_words = df_words.reset_index()


df_pat_aggregated = df_pat.drop(["Patient"], axis=1)
df_pat_aggregated = df_pat_aggregated.groupby('Class').agg('mean')
df_pat_aggregated = df_pat_aggregated.reset_index()
df_pat_aggregated["words"]=df_words["words"]
os.chdir(r"C:\Users\clalk\JLUbox\Transkriptanalysen\3 KOGNITIVE VERZERRUNGEN\Analysen")

df_pat_aggregated.to_excel('ergebnisse_kognitive_verzerrungen_mean_5_break.xlsx')
