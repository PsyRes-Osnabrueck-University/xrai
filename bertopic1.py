from HanTa import HanoverTagger as ht
import re
import os
# Create path and sub folders
base_path = "C:/Users/Christopher/JLUbox/Transkriptanalysen/2 TOPIC MODELING/Analysen/"
sub_folder_processing = "data/processing"
sub_folder_Test_transkripte = "data/Transkripte/Test-transkripte"
sub_folder_data = "data"
sub_folder_output = "output"

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import requests
import tensorflow as tf
import time
import ast

from sklearn.feature_extraction.text import CountVectorizer

from bertopic import BERTopic
import transformers
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

hannover = ht.HanoverTagger('morphmodel_ger.pgz')

import sys
sys.path.insert(0,r'C:\Users\Christopher\PycharmProjects\Bertopic\topictuner')
from topictuner import TopicModelTuner as TMT






# Load the transformer model and tokenizer for gbert-large
tokenizer = transformers.AutoTokenizer.from_pretrained("deepset/gbert-large")
model = AutoModel.from_pretrained("deepset/gbert-large")


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# create embeddings
def create_embeddings(paragraph_list):
    encoded_input = tokenizer(paragraph_list, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    sentence_embeddings_arr = sentence_embeddings.numpy()
    return sentence_embeddings_arr


def rm_stop_wds(text):
  for word in stop_words:
    word = re.compile(word, re.IGNORECASE)
    text = re.sub(word, "", text)
  return text

def clean_text(text):
    RE_WSPACE = re.compile(r"\s+", re.IGNORECASE)
    RE_TAGS = re.compile(r"<[^>]+>")
    RE_ASCII = re.compile(r"[^A-Za-zÀ-ž,.!? ]", re.IGNORECASE)
    RE_SINGLECHAR = re.compile(r"\b[A-Za-zÀ-ž,.!?]\b", re.IGNORECASE)
    # Sonderzeichen auswählen

    text = rm_stop_wds(text)
    # Stopwords entfernen

    text = re.sub(RE_TAGS, " ", text)
    text = re.sub(RE_ASCII, " ", text)
    text = re.sub(RE_SINGLECHAR, " ", text)
    text = re.sub(RE_WSPACE, " ", text)
    # Sonderzeichen löschen

    return text

# open("stop_words_short.txt")
path = os.path.join(base_path,sub_folder_processing)
os.chdir(path)
os.listdir()
stop_words = []
with open("stop_words_short.txt", 'r', encoding="utf8") as f:
    for line in f:
        stop_words.append(r'\b' + line.rstrip("\n") + r'\b') #\b ist wichtig, damit nur ganze Wörter entfernt werden

path = os.path.join(base_path,sub_folder_Test_transkripte)
file_list = os.listdir(path)
os.chdir(path)
print(file_list)

# Dataframe mit Transkripten erstellen
min_num_words = 5
df_fin = pd.DataFrame(columns=['File', 'Therapist_no', 'Patient_no', 'Session', 'Therapist', 'Patient'])
i=0
for filename in file_list:
  with open(filename, "r",encoding = 'unicode_escape') as input_file:
    transcript_raw = input_file.readlines()
    transcript_clean=[]
    for paragraph in transcript_raw:
      paragraph_clean = clean_text(paragraph)  # säubern
      if not len(paragraph_clean.split())<min_num_words:    # kurze Absätze aussortieren
        transcript_clean.append(paragraph_clean)
    #transcript_clean = ' '.join(transcript_clean)
    # Transkripte in Dataframe speichern:
    sfn = filename.replace("therapeut.", "")
    sfn = sfn.replace("patient.", "")
    if "therapeut." in filename:
      if sfn in df_fin["File"].tolist():
        df_fin.at[df_fin["File"].tolist().index(sfn),'Therapist'] = transcript_clean
      if not sfn in df_fin["File"].tolist():
        df_fin.loc[i] = [sfn] + [sfn[0:4]] + [sfn[5:7]] + [sfn[8:10]] + [transcript_clean] + [""]
        i += 1
    elif "patient." in filename:
      if sfn in df_fin["File"].tolist():
        df_fin.at[df_fin["File"].tolist().index(sfn),'Patient'] = transcript_clean
      if not sfn in df_fin["File"].tolist():
        df_fin.loc[i] = [sfn] + [sfn[0:4]] + [sfn[5:7]] + [sfn[8:10]] + [""] + [transcript_clean]
        i += 1
# Speichern des df_fin
path = os.path.join(base_path,sub_folder_data)
os.chdir(path)
df_fin.to_json('Transkripttabelle_test.json', orient='split')
df_fin = pd.read_json('Transkripttabelle_test.json',orient='split')

#Patientencorpus

df_pat = pd.DataFrame(columns=['Class', 'Patient'])
dict_pat = []
for i in range(len(df_fin)):
    for paragraph in df_fin["Patient"][i]:
        new_row = {'Class': df_fin["File"][i], 'Patient': paragraph}
        dict_pat.append(new_row)
    print(i)
df_pat = pd.DataFrame.from_dict(dict_pat)


list_pat = df_pat["Patient"].tolist()
classes_pat = df_pat["Class"]

# Patientencorpus speichern und laden
path = os.path.join(base_path,sub_folder_data)
os.chdir(path)
df_pat.to_json('Patientenabsätze_test.json', orient='split')
df_pat = pd.read_json('Patientenabsätze_test.json',orient='split')



# Topic Tuner
tmt = TMT()
# Achtung: Dauert lange
begin_time = time.time()
embeddings_array = create_embeddings(df_pat["Patient"].tolist()[0:10000])
end_time = time.time()
interval_time = end_time - begin_time
print(interval_time/60)


tmt.docs = df_pat["Patient"].tolist()[0:100]
tmt.reduce()

lastRunResultsDF = tmt.randomSearch([*range(2,30)], [.1, .25, .5, .75, 1])
fig = tmt.visualizeSearch(lastRunResultsDF)
fig.show(renderer="browser")
tmt.summarizeResults(lastRunResultsDF).sort_values(by=['number_uncategorized'])
lastRunResultsDF = tmt.randomSearch([*range(20,75)], [.1, .2, .3], iters = 50)

lastRunResultsDF = tmt.gridSearch([*range(40,70)], [.1, .2, .3, .4, .5, .6, .7])

csizes = []
ssizes = []
for csize in range(2,30) :
  for ssize in range(1, csize+1) :
    csizes.append(csize)
    ssizes.append(ssize)
lastRunResultsDF = tmt.simpleSearch(csizes, ssizes)

tmt.save('temp')

tmt.docs = df_pat["Patient"].tolist()[0:100]
bt1 = tmt.getBERTopicModel(3,1)
bt1.fit_transform(tmt.docs, tmt.embeddings)
bt1.get_topic_info()



tmt.createVizReduction()
fig2 = tmt.visualizeEmbeddings(33,8)
fig2.show(renderer="browser")

tmt = TMT.wrapBERTopicModel(Berti)

Berti = BERTopic(embedding_model=multilingual, top_n_words = 10, nr_topics="auto", n_gram_range = (1,3), calculate_probabilities=False, diversity=0.2, verbose = True)

topics, probabilities = Berti.fit_transform(list_pat)
path = base_path
os.chdir(path)
model.save("my_model")
model = BERTopic.load("my_model")

# Remove additional stopwords
vectorizer_model = CountVectorizer(stop_words=frozenset(["er", "sie","ihn", "ihr", "ihm", "sagt", "gesagt", "der", "die"]), ngram_range=(1, 3))
model.update_topics(list_pat, vectorizer_model=vectorizer_model)
path = base_path
os.chdir(path)
model.save("model_sw_rm")
model = BERTopic.load("model_sw_rm")

model.get_topic(9)

model_div = model # duplicate Model

# diversify model




topic_model = BERTopic(embedding_model=sentence_model, diversity=0.2)


div = BERTopic(diversity=0.2)
model_div.update_topics(list_pat, vectorizer_model=div)
model_div.get_topic(9)

# Save and load Patient-model
path = os.path.join(base_path,sub_folder_output)
os.chdir(path)
model_div.save("Patient_model")
model_div = BERTopic.load("Patient_model")




# topics per class
topics_per_class = model.topics_per_class(list_pat, classes=classes_pat)

topics_per_class.to_excel("topics_per_class.xlsx")

figure = model.visualize_topics_per_class(topics_per_class, top_n_topics=10)
figure.show() # krieg show grad nicht zum Laufen

#Therapeutencorpus
df_ther = pd.DataFrame(columns=['Class', 'Therapist'])
for i in range(len(df_fin)):
    for paragraph in df_fin["Therapist"][i]:
        new_row = {'Class': df_fin["File"][i], 'Patient': paragraph}
        df_ther = df_ther.append(new_row, ignore_index=True)
    print(i)

list_ther = df_ther["Therapist"]

model = BERTopic(language="german", top_n_words = 10, nr_topics="auto", n_gram_range = (1,3), calculate_probabilities=True, verbose = True)

topics, probabilities = model.fit_transform(list_ther)



model.visualize_topics()

model.visualize_barchart()

model.visualize_heatmap()

new_topics, new_probs = model.reduce_topics(corpus_lemma, topics, probabilities, nr_topics=465)

model.get_topic_freq()

model.visualize_topics()

model.visualize_barchart()

model.get_topic(10)

for x in range(0, 100):
    first_tuple_elements = []
    for tuple in model.get_topic(x):
      first_tuple_elements.append(tuple[0])
    print(first_tuple_elements)
    print("\n")

model.visualize_hierarchy(top_n_topics=20)

new_topics, new_probs = model.reduce_topics(corpus_lemma, topics, probabilities, nr_topics=200)

for x in range(0, 10):
    first_tuple_elements = []
    for tuple in model.get_topic(x):
      first_tuple_elements.append(tuple[0])
    print(first_tuple_elements)
    print("\n")

data = pd.read_csv(r"/content/dataFramedynamic.csv")
data1 = pd.read_csv(r"/content/patient.csv")


docs = data['text']
targets = data['doc_id']
target_names = data1['pat_code']
classes = [data1['pat_code'][i] for i in data['doc_id']]

topic_model = BERTopic(verbose=True)
topics, probs = topic_model.fit_transform(docs)

topics_per_class = topic_model.topics_per_class(docs, topics, classes=classes)

topic_model.visualize_topics_per_class(topics_per_class)

timestamps = data['session_no']
texts = data['text']

topic_model = BERTopic(verbose=True)
topics, probs = topic_model.fit_transform(texts)

topics_over_time = topic_model.topics_over_time(texts, topics, timestamps)

topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=20)
#session Anzahl: [00:8, 01:6, 03:76, 04:12, 05:46, 
#06:11, 07:9, 08:5, 09:9, 10:43, 
#11:10, 12:8, 13:4, 14:5, 14.15:1, 
#15:46, 16:4, 17:8, 18:4, 19:8, 
#20:34, 21:2, 22:6, 23:2, 24:2, 
#25:46, 26:3, 27:2, 28:1, 29:4, 
#30:11, 31:2, 32:2, 33:1, 34:1, 
#35:2, 37:1, 40:3, 42:1, 45:3, 
#47:1, 48:2 49:1, 50:2, 52:1, 
#55:1, 57:1, 60:1, 62:1, 65:1, 85:1]

data_sub = pd.read_csv(r"/content/dataFrame1012P08.csv")
timestamps = data_sub['session_no']
texts = data_sub['text']

topic_model = BERTopic(verbose=True)
topics, probs = topic_model.fit_transform(texts)

topics_over_time = topic_model.topics_over_time(texts, topics, timestamps)

topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=10)
