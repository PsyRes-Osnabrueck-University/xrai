## Lade die benötigten Packages
import os
import pandas as pd
import numpy as np
import re
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

df_pat = pd.read_excel("Patientenabsätze vollständig.xlsx")

## Kleinschreibung für df_pat
df_pat["Patient"] = df_pat["Patient"].str.lower()

## lese das Wörterbuch ein
kognitive_verzerrungen = pd.read_excel(r"C:\Users\clalk\JLUbox\Transkriptanalysen\3 N-GRAMME\Wörterbuch_deutsch_fin.xlsx")
kognitive_verzerrungen = kognitive_verzerrungen.rename(columns={'Unnamed: 2': 'Verzerrung 2', 'Unnamed: 3': 'Verzerrung 3','Unnamed: 4': 'Verzerrung 4', 'Unnamed: 5': 'Verzerrung 5', 'Unnamed: 6': 'Verzerrung 6', 'Unnamed: 7': 'Verzerrung 7', 'Unnamed: 8': 'Verzerrung 8', 'Unnamed: 9': 'Verzerrung 9', 'Unnamed: 10': 'Verzerrung 10'})
# Kleinschreibung auf kognitive_verzerrungen anwenden
kognitive_verzerrungen = kognitive_verzerrungen.applymap(lambda s:s.lower() if type(s) == str else s)

# Kognitive Verzerrungen: in ein Dataframe mit nur zwei Spalten übertragen, damit Auswertung leichter ist.Schleife durchführen und Werte in den neuen Dataframe übertragen
# Neue Dataframe erstellen
new_kognitive_verzerrungen = pd.DataFrame(columns=['Column 1', 'Column 2'])

# Schleife durchführen und Werte in den neuen Dataframe übertragen
for index, row in kognitive_verzerrungen.iterrows():
    new_rows = []
    for i in range(1, len(row)):
        new_rows.append({'Column 1': row[0], 'Column 2': row[i]})
    new_kognitive_verzerrungen = pd.concat([new_kognitive_verzerrungen, pd.DataFrame(new_rows)], ignore_index=True)

# Drop NAs
new_kognitive_verzerrungen = new_kognitive_verzerrungen.dropna(subset=['Column 2'])

# Drop Duplicates (Spalte 2)
new_kognitive_verzerrungen = new_kognitive_verzerrungen.drop_duplicates(subset=['Column 2'])

## Erstelle ein neues Dataframe. In dieses kommen die Ergebnisse der kognitiven Verzerrungen.
### in der ersten Spalte sind verschiedene Einträge der ersten Spalte aus df_pat
new_df_pat = pd.DataFrame(df_pat["Class"].drop_duplicates())

# Schleife durch die einzigartigen Werte in der Spalte "Kategorie" um die verschiedenen kognitiven Verzerrungen hinzuzufügen.
for category in kognitive_verzerrungen["Kategorie"].unique():

    # Überprüfe, ob der Spaltenname schon im DataFrame existiert
    if category not in new_df_pat.columns:
        # Füge eine neue Spalte hinzu
        new_df_pat[category] = None

## fülle new_df_pat mit Nullen
new_df_pat = new_df_pat.fillna(0)
new_df_pat = new_df_pat.set_index("Class")

'''
## hier werden Testdaten erstellt, um den Code zu überprüfen, damit nicht so lange gewartet werden muss.
df_pat = df_pat.iloc[:500, :]
df_pat.at[0,"Patient"] = "ich bin ein"
df_pat.at[1,"Patient"]  = "ich bin eine"
df_pat.at[2,"Patient"] = "er ist ein"
df_pat.at[3,"Patient"] = "schlimmster"
df_pat.at[400,"Patient"] = "fatalster"
'''

### prüfe, ob die kognitiven Verzerrungen in den Patientenabsätzen vorkommen
# outer loop
for i in range(0, len(df_pat)):
    print(i)
# inner loop
    for j in range(0, len(new_kognitive_verzerrungen)):
        if new_kognitive_verzerrungen.iloc[j]["Column 2"] in df_pat.iloc[i]["Patient"]:
            x = df_pat.iloc[i]["Class"]  # das ist der Name des Transkriptes
            y = new_kognitive_verzerrungen.iloc[j]["Column 1"] # das ist der Name der Kategorie der erkannten kognitiven Verzerrung
            new_df_pat.loc[x][y] = new_df_pat.loc[x][y] + 1
            break



# Ergänze 1. Person Singular Personalpronomen in new_df_pat
## Erstelle Vektor mit Personalpronomen 1. Person Singular
personalpronomen_1_singular = "ich", "mein", "meine", "meiner", "meines", "meins", "mir", "mich"
personalpronomen_1_singular = pd.DataFrame({'Spalte 1': personalpronomen_1_singular})
## Füge Spalte hinzu
new_df_pat["1. person singular personalpronomen"] = None
new_df_pat["ich"] = None
new_df_pat["mein"] = None
new_df_pat["meine"] = None
new_df_pat["meiner"] = None
new_df_pat["meines"] = None
new_df_pat["meins"] = None
new_df_pat["mir"] = None
new_df_pat["mich"] = None

## fülle new_df_pat mit Nullen
new_df_pat = new_df_pat.fillna(0)
new_df_pat = new_df_pat.set_index("Class")

### prüfe, ob die 1. Person Singular Personalpronomen in den Patientenabsätzen vorkommen
# outer loop
for i in range(0, len(df_pat)):
    print(i)
# inner loop
    for j in range(0, len(personalpronomen_1_singular)):
        if personalpronomen_1_singular.iloc[j]["Spalte 1"] in df_pat.iloc[i]["Patient"]:
            count = df_pat.iloc[i]["Patient"].count(personalpronomen_1_singular.iloc[j]["Spalte 1"])# hier ist die Anzahl, wie häufig personalpronomen_1_singular.iloc[j]["Spalte 1"] in df_pat.iloc[i]["Patient"] vorkommt.
            x = df_pat.iloc[i]["Class"]  # das ist der Name des Transkriptes
            new_df_pat.loc[x]["1. person singular personalpronomen"] = new_df_pat.loc[x]["1. person singular personalpronomen"] + count

# Speichern des Dataframes als JSON-Datei
os.chdir(r"C:\Users\clalk\JLUbox\Transkriptanalysen\3 N-GRAMME\Analysen")
new_df_pat.to_json('ergebnisse_kognitive_verzerrungen.json')
new_df_pat.to_excel('ergebnisse_kognitive_verzerrungen.xlsx')

