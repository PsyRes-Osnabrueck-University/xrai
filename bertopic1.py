from HanTa import HanoverTagger as ht
import re
import os
import json
import pyreadstat
# Create path and sub folders
base_path = "C:/Users/Christopher/JLUbox/Transkriptanalysen/2 TOPIC MODELING/Analysen/"
sub_folder_processing = "data/processing"
sub_folder_transkripte = "data/Transkripte"
sub_folder_data = "data"
sub_folder_output = "output"

# base_path = "C:/Users/Christopher/PycharmProjects/BerTopic/"
base_path_pycharm = "C:/Users/JLU-SU/PycharmProjects/BerTopic/"

import nltk
from nltk.corpus import stopwords
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import requests
import tensorflow as tf
import time
import ast

from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer

from bertopic import BERTopic
import transformers
from transformers import AutoConfig, AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F

hannover = ht.HanoverTagger('morphmodel_ger.pgz')

import sys
sys.path.insert(0,r'C:\Users\Christopher\PycharmProjects\Bertopic\TopicTuner')
from topictuner import TopicModelTuner as TMT
from hdbscan import HDBSCAN

# From here on: https://discuss.huggingface.co/t/make-bert-inference-faster/9930/4


# Load the transformer model and tokenizer for gbert-large
tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-german-cased")
model = AutoModel.from_pretrained("bert-base-german-cased")

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
    text = text.lstrip("?!.:,; ")
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

path = os.path.join(base_path,sub_folder_transkripte)
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
# df_fin.to_json('Transkripttabelle_fertig.json', orient='split') - bitte nicht mehr überschreiben!!!
# df_fin.to_json('Transkripttabelle_fertig_10.json', orient='split') #- bitte nicht mehr überschreiben!!!
# df_fin = pd.read_json('Transkripttabelle_fertig_5.json',orient='split')
# df_fin = pd.read_json('Transkripttabelle_fertig_10.json',orient='split')
#df_fin = pd.read_json('Transkripttabelle_test_5.json',orient='split')

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
# df_pat.to_json('Patientenabsätze_fertig.json', orient='split')
# df_pat.to_json('Patientenabsätze_fertig_10.json', orient='split')

# df_pat = pd.read_json('Patientenabsätze_fertig_5.json',orient='split')
# df_pat = pd.read_json('Patientenabsätze_fertig_10.json',orient='split')


model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
sentences = df_pat["Patient"].tolist()
begin_time = time.time()
embeddings = model.encode(sentences)
end_time = time.time()
interval_time = end_time - begin_time
print(interval_time)


####### Topic Tuning ausklammern

tmt = TMT()
tmt.embeddings = embeddings
tmt.docs = df_pat["Patient"].tolist()

path = os.path.join(base_path,sub_folder_data)
os.chdir(path)
# tmt.save('five_docs')
# tmt = tmt.load('five_docs')
# tmt.save('ten_docs')
# tmt = tmt.load('ten_docs')



df_test = tmt.summarizeResults(lastRunResultsDF)
df_neu = df_test.sort_values(by=['number_uncategorized'])
print(df_neu)
lastRunResultsDF = tmt.pseudoGridSearch([*range(130,160)], [x/100 for x in range(70,100,5)])
fig = tmt.visualizeSearch(lastRunResultsDF)
fig.show(renderer="browser")

# für 79, 67 kommen wir bei pat_10 auf 14.7k missing topics

# 2-d visualization:
tmt2 = tmt
tmt2.createVizReduction('TSNE')
tmt2.visualizeEmbeddings(149, 141).show(renderer="browser")

tmt.bestParams = (133, 93) # This sets default parameters - not necessary, see next
BERTopic_model = tmt.getBERTopicModel() # this will use the bestParams, but you could also just getBERTopicModel(5,5)
ctfidf_model = ClassTfidfTransformer()


BERTopic_model.vectorizer_model = cv
BERTopic_model.ctfidf_model = ctfidf_model
BERTopic_model.calculate_probabilities = True

topics, probabilities = BERTopic_model.fit_transform(tmt.docs, embeddings=tmt.embeddings)

####### Ende Klammer TopicTuner


######## BERTOPIC Variation number_of_topics = [150, 200, 250]



nltk.download('stopwords')

stops = set(stopwords.words('german'))
stops = stops.union(frozenset(["schon", "halt","mal", "ja", "ne"]))
# I like to use trigrams as well as bigrams - but it could be 1,2 etc.
cv = CountVectorizer(ngram_range=(1, 3), stop_words=stops)



# Stopwords
nltk.download('stopwords')

stops = set(stopwords.words('german'))
stops = stops.union(frozenset(["schon", "halt","mal", "ja", "ne"]))
# I like to use trigrams as well as bigrams - but it could be 1,2 etc.
cv = CountVectorizer(ngram_range=(1, 3), stop_words=stops)


# BERTopic Model
model = BERTopic(language="German", vectorizer_model = cv, top_n_words = 10, nr_topics=250, n_gram_range = (1,3), calculate_probabilities=True, verbose = True)
topics, probabilities = model.fit_transform(sentences, embeddings=embeddings)


# BERTopic angucken
df_topic_overview = BERTopic_model.get_document_info(tmt.docs)
df_topics_pat_5_200 = model.get_topic_info()
df_topic_overview = df_topic_overview.sort_values(by=['Topic'])
path = os.path.join(base_path,sub_folder_data)
os.chdir(path)
df_topics_pat_5_150.to_excel("topics_pat_5_150.xlsx")

# BERTopic speichern
path = os.path.join(base_path,sub_folder_data)
os.chdir(path)
model.save("Pat_5words_200")
# model2 = BERTopic.load("Pat_5words_200")


# BERTopic Model pat_10words_150
#################################################################################################################################
# Embeddings für min. 10 Wörter
df_pat = pd.read_json('Patientenabsätze_fertig_10.json',orient='split')
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
sentences = df_pat["Patient"].tolist()
begin_time = time.time()
embeddings = model.encode(sentences)
end_time = time.time()
interval_time = end_time - begin_time
print(interval_time)

#BERTopic model
model = BERTopic(language="German", vectorizer_model = cv, top_n_words = 10, nr_topics=150, n_gram_range = (1,3), calculate_probabilities=True, verbose = True)
topics, probabilities = model.fit_transform(sentences, embeddings=embeddings)


# Topics speichern
path = os.path.join(base_path,sub_folder_data)
os.chdir(path)
df_topics_pat_10_150 = model.get_topic_info()
df_topics_pat_10_150.to_excel("topics_pat_10_150.xlsx")

# BERTopic speichern
path = os.path.join(base_path,sub_folder_data)
os.chdir(path)
model.save("Pat_10words_150")


# BERTopic Model pat_10words_200
#################################################################################################################################
# Embeddings wurden erstellt!


#BERTopic model
model = BERTopic(language="German", vectorizer_model = cv, top_n_words = 10, nr_topics=200, n_gram_range = (1,3), calculate_probabilities=True, verbose = True)
topics, probabilities = model.fit_transform(sentences, embeddings=embeddings)


# Topics speichern
path = os.path.join(base_path,sub_folder_data)
os.chdir(path)
df_topics_pat_10_200 = model.get_topic_info()
df_topics_pat_10_200.to_excel("topics_pat_10_200.xlsx")

# BERTopic speichern
path = os.path.join(base_path,sub_folder_data)
os.chdir(path)
model.save("Pat_10words_200")

# BERTopic Model pat_10words_250
#################################################################################################################################
# Embeddings wurden erstellt!


#BERTopic model
model = BERTopic(language="German", vectorizer_model = cv, top_n_words = 10, nr_topics=250, n_gram_range = (1,3), calculate_probabilities=True, verbose = True)
topics, probabilities = model.fit_transform(sentences, embeddings=embeddings)


# Topics speichern
path = os.path.join(base_path,sub_folder_data)
os.chdir(path)
df_topics_pat_10_250 = model.get_topic_info()
df_topics_pat_10_250.to_excel("topics_pat_10_250.xlsx")

# BERTopic speichern
path = os.path.join(base_path,sub_folder_data)
os.chdir(path)
model.save("Pat_10words_250")


# BERTopic Model ther_5words_150
#################################################################################################################################
# Embeddings erstellen!
df_ther = pd.read_json('Therapeutenabsätze_fertig_5.json',orient='split')
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
sentences = df_ther["Therapist"].tolist()
begin_time = time.time()
embeddings = model.encode(sentences)
end_time = time.time()
interval_time = end_time - begin_time
print(interval_time)


#BERTopic model
model = BERTopic(language="German", vectorizer_model = cv, top_n_words = 10, nr_topics=150, n_gram_range = (1,3), calculate_probabilities=True, verbose = True)
topics, probabilities = model.fit_transform(sentences, embeddings=embeddings)


# Topics speichern
path = os.path.join(base_path,sub_folder_data)
os.chdir(path)
df_topics_ther_5_150 = model.get_topic_info()
df_topics_ther_5_150.to_excel("topics_ther_5_150.xlsx")

# BERTopic speichern
path = os.path.join(base_path,sub_folder_data)
os.chdir(path)
model.save("Ther_5words_150")


# BERTopic Model ther_5words_200
#################################################################################################################################
# Embeddings fertig!



#BERTopic model
model = BERTopic(language="German", vectorizer_model = cv, top_n_words = 10, nr_topics=200, n_gram_range = (1,3), calculate_probabilities=True, verbose = True)
topics, probabilities = model.fit_transform(sentences, embeddings=embeddings)


# Topics speichern
path = os.path.join(base_path,sub_folder_data)
os.chdir(path)
df_topics_ther_5_200 = model.get_topic_info()
df_topics_ther_5_200.to_excel("topics_ther_5_200.xlsx")

# BERTopic speichern
path = os.path.join(base_path,sub_folder_data)
os.chdir(path)
model.save("Ther_5words_200")


# BERTopic Model ther_5words_250
#################################################################################################################################
# Embeddings fertig!



#BERTopic model
model = BERTopic(language="German", vectorizer_model = cv, top_n_words = 10, nr_topics=250, n_gram_range = (1,3), calculate_probabilities=True, verbose = True)
topics, probabilities = model.fit_transform(sentences, embeddings=embeddings)


# Topics speichern
path = os.path.join(base_path,sub_folder_data)
os.chdir(path)
df_topics_ther_5_250 = model.get_topic_info()
df_topics_ther_5_250.to_excel("topics_ther_5_250.xlsx")

# BERTopic speichern
path = os.path.join(base_path,sub_folder_data)
os.chdir(path)
model.save("Ther_5words_250")

# BERTopic Model ther_10words_150
#################################################################################################################################
# Embeddings erstellen!
df_ther = pd.read_json('Therapeutenabsätze_fertig_10.json',orient='split')
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
sentences = df_ther["Therapist"].tolist()
begin_time = time.time()
embeddings = model.encode(sentences)
end_time = time.time()
interval_time = end_time - begin_time
print(interval_time)

csizes = []
ssizes = []
for csize in range(2,30) :
  for ssize in range(1, csize+1) :
    csizes.append(csize)
    ssizes.append(ssize)
lastRunResultsDF = tmt.simpleSearch(csizes, ssizes)

#BERTopic model
model = BERTopic(language="German", vectorizer_model = cv, top_n_words = 10, nr_topics=150, n_gram_range = (1,3), calculate_probabilities=True, verbose = True)
topics, probabilities = model.fit_transform(sentences, embeddings=embeddings)

tmt.docs = df_pat["Patient"].tolist()[0:100]
bt1 = tmt.getBERTopicModel(3,1)
bt1.fit_transform(tmt.docs, tmt.embeddings)
bt1.get_topic_info()

# Topics speichern
path = os.path.join(base_path,sub_folder_data)
os.chdir(path)
df_topics_ther_10_150 = model.get_topic_info()
df_topics_ther_10_150.to_excel("topics_ther_10_150.xlsx")

# BERTopic speichern
path = os.path.join(base_path,sub_folder_data)
os.chdir(path)
model.save("Ther_10words_150")

tmt.createVizReduction()
fig2 = tmt.visualizeEmbeddings(33,8)
fig2.show(renderer="browser")

# BERTopic Model ther_10words_200
#################################################################################################################################
# Embeddings sind fertig!

Berti = BERTopic(embedding_model=multilingual, top_n_words = 10, nr_topics="auto", n_gram_range = (1,3), calculate_probabilities=False, diversity=0.2, verbose = True)

#BERTopic model
model = BERTopic(language="German", vectorizer_model = cv, top_n_words = 10, nr_topics=200, n_gram_range = (1,3), calculate_probabilities=True, verbose = True)
topics, probabilities = model.fit_transform(sentences, embeddings=embeddings)


# Topics speichern
path = os.path.join(base_path,sub_folder_data)
os.chdir(path)
df_topics_ther_10_200 = model.get_topic_info()
df_topics_ther_10_200.to_excel("topics_ther_10_200.xlsx")

# BERTopic speichern
path = os.path.join(base_path,sub_folder_data)
os.chdir(path)
model.save("Ther_10words_200")

# BERTopic Model ther_10words_250
#################################################################################################################################
# Embeddings sind fertig!


#BERTopic model
model = BERTopic(language="German", vectorizer_model = cv, top_n_words = 10, nr_topics=250, n_gram_range = (1,3), calculate_probabilities=True, verbose = True)
topics, probabilities = model.fit_transform(sentences, embeddings=embeddings)


# Topics speichern
path = os.path.join(base_path,sub_folder_data)
os.chdir(path)
df_topics_ther_10_250 = model.get_topic_info()
df_topics_ther_10_250.to_excel("topics_ther_10_250.xlsx")

# BERTopic speichern
path = os.path.join(base_path,sub_folder_data)
os.chdir(path)
model.save("Ther_10words_250")



# Therapeutenkorpus
df_ther = pd.DataFrame(columns=['Class', 'Therapist'])
dict_ther = []
for i in range(len(df_fin)):
    for paragraph in df_fin["Therapist"][i]:
        new_row = {'Class': df_fin["File"][i], 'Therapist': paragraph}
        dict_ther.append(new_row)
    print(i)
df_ther = pd.DataFrame.from_dict(dict_ther)



# Therapeutenkorpus speichern und laden
path = os.path.join(base_path,sub_folder_data)
os.chdir(path)
# df_ther.to_json('Therapeutenabsätze_fertig_10.json', orient='split')
# df_ther.to_json('Therapeutenabsätze_fertig_5.json', orient='split')

# df_ther = pd.read_json('Therapeutenabsätze_fertig_10.json',orient='split')
# df_ther = pd.read_json('Therapeutenabsätze_fertig_5.json',orient='split')


path = os.path.join(base_path,sub_folder_data)
os.chdir(path)


####### Topic Tuning ausklammern
tmt = TMT()
tmt.embeddings = embeddings
tmt.docs = df_ther["Therapist"].tolist()

path = os.path.join(base_path,sub_folder_data)
os.chdir(path)
# tmt.save('five_docs_ther')
# tmt = tmt.load('five_docs_ther')
# tmt.save('ten_docs_ther')
# tmt = tmt.load('ten_docs_ther')


lastRunResultsDF = tmt.randomSearch([2,3,4,5], [.1, .1, .2, .4])
fig = tmt.visualizeSearch(lastRunResultsDF)
fig.show(renderer="browser")

df_test = tmt.summarizeResults(lastRunResultsDF)
df_neu = df_test.sort_values(by=['number_uncategorized'])
print(df_neu)
lastRunResultsDF = tmt.pseudoGridSearch([*range(130,160)], [x/100 for x in range(70,100,5)])
fig = tmt.visualizeSearch(lastRunResultsDF)
fig.show(renderer="browser")

# für 79, 67 kommen wir bei pat_10 auf 14.7k missing topics

# 2-d visualization:
tmt2 = tmt
tmt2.createVizReduction('TSNE')
tmt2.visualizeEmbeddings(149, 141).show(renderer="browser")

tmt.bestParams = (133, 93) # This sets default parameters - not necessary, see next



BERTopic_model = tmt.getBERTopicModel() # this will use the bestParams, but you could also just getBERTopicModel(5,5)
ctfidf_model = ClassTfidfTransformer()


####### Ende Klammer TopicTuner


######## BERTOPIC Variation number_of_topics = [150, 200, 250]

# Stopwords
nltk.download('stopwords')

stops = set(stopwords.words('german'))
stops = stops.union(frozenset(["schon", "halt","mal", "ja", "ne"]))
# I like to use trigrams as well as bigrams - but it could be 1,2 etc.
cv = CountVectorizer(ngram_range=(1, 3), stop_words=stops)


# Embeddings berechnen
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
#sentences = df_ther["Therapist"].tolist()
sentences = df_pat["Patient"].tolist()
begin_time = time.time()
embeddings = model.encode(sentences)
end_time = time.time()
interval_time = end_time - begin_time
print(interval_time)

# BERTopic Model
model = BERTopic(language="German", vectorizer_model = cv, top_n_words = 10, nr_topics=150, n_gram_range = (1,3), calculate_probabilities=True, verbose = True)
topics, probabilities = model.fit_transform(sentences, embeddings=embeddings)

# BERTopic angucken
df_topic_overview = BERTopic_model.get_document_info(tmt.docs)
df_topics = model.get_topic_info()
df_topic_overview = df_topic_overview.sort_values(by=['Topic'])

# BERTopic speichern
path = os.path.join(base_path,sub_folder_data)
os.chdir(path)
model.save("Pat_10words_150")
model = BERTopic.load("Pat_10words_150")




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




# Topic-Document-Matrix erstellen
topic_document_probabilities = pd.DataFrame(model.probabilities_)
topic_document_matrix = pd.concat([df_pat, topic_document_probabilities], axis =1 )
topic_labels = [i for i in model.topic_labels_.values()]
del topic_labels[0]
topic_document_matrix.columns = ["Class", "Patient"] + topic_labels
#Aggregiere die Summen der Paragrafen zu einem Transkript
cols_to_sum = topic_document_matrix.columns[2:]
topic_document_matrix_sum = topic_document_matrix.groupby('Class')[cols_to_sum].sum()

# Füge den Sitzungscode ein
session_list = [i[:-4] for i in topic_document_matrix_sum.index] # Dateiendung entfernen
session_list = [i.replace(" (fertig)", "") for i in session_list] # Fertig entfernen
session_list = [i.replace("(fertig)", "") for i in session_list] # ""
session_list = [i.replace("u15", "") for i in session_list] # u15-Endung entfernen
topic_document_matrix_sum.insert(0, column="session", value=session_list) # Spalte Session an Pos. 0 erstellen

## lese SPSS Daten ein
path = os.path.join(base_path,sub_folder_processing)
os.chdir(path)
sitzungsbogen = pd.read_spss('Stundenbögen_20190810.sav')
sitzungsbogen["session"] = sitzungsbogen["CODE"] + "_" + sitzungsbogen["INSTANCE"]
topic_document_matrix_sum["hscl"]=0 # Spalte hscl erstellen

### ab hier werden die hscl-scores in topic_document_matrix_sum eingetragen
for i in range(len(topic_document_matrix_sum)):
    print(i)
    session=topic_document_matrix_sum.iloc[[i]]["session"][0]

##### Dieser Part ist wichtig, da so die HSCL-Scores der nächsten Sitzung (nicht der aktuellen) vorhergesagt werden
    try:
        number = int(session[-2:])
        number += 1
        number_str = str(number).zfill(2)
        next_session = session[:-2] + number_str
    except ValueError:
        next_session = session
    print(next_session)
    session = next_session
    try: # Versuche den hscl zu übergeben
        hscl = sitzungsbogen[sitzungsbogen['session'].str.match(session)]["Gesamtscore_hscl"].iloc[0]
        topic_document_matrix_sum.iloc[i, -1] = hscl # die -1 steht für die hinterste Spalte. Muss ggbfalls angepasst werden, zb -2 für vorletzte Spalte
    except Exception: # falls er ihn nicht findet, den Fehler ignorieren und NA eintragen.
        topic_document_matrix_sum.iloc[i, -1] = "NA" # s.o.

### ab hier werden die srs_ges-scores in topic_document_matrix_sum eingetragen
topic_document_matrix_sum["srs_ges"]=0 # Spalte srs_ges erstellen

for i in range(len(topic_document_matrix_sum)):
    print(i)
    session=topic_document_matrix_sum.iloc[[i]]["session"][0]

    try: # Versuche den srs_ges zu übergeben
        srs_ges = sitzungsbogen[sitzungsbogen['session'].str.match(session)]["srs_ges"].iloc[0]
        topic_document_matrix_sum.iloc[i, -1] = srs_ges # die -1 steht für die hinterste Spalte. Muss ggbfalls angepasst werden, zb -2 für vorletzte Spalte
    except Exception: # falls er ihn nicht findet, den Fehler ignorieren und NA eintragen.
      topic_document_matrix_sum.iloc[i, -1] = "NA" # s.o.
        ### -> 16 NA/nan

### ab hier werden die aktuellen hscl_scores in topic_document_matrix_sum eingetragen
#### vorher muss ggf. der Sitzungsbogen und topic_document_outcome_patient_5_250.xlsx bzw. topic_document_outcome_therapeut_5_250 eingelesen werden
topic_document_matrix_sum["hscl_aktuelle_sitzung"]=0 # Spalte hscl erstellen

for i in range(len(topic_document_matrix_sum)):
    print(i)
    session=topic_document_matrix_sum.iloc[[i]]["session"][0]

    try: # Versuche den HSCL zu übergeben
        hscl_aktuelle_sitzung = sitzungsbogen[sitzungsbogen['session'].str.match(session)]["Gesamtscore_hscl"].iloc[0]
        topic_document_matrix_sum.iloc[i, -1] = hscl_aktuelle_sitzung # die -1 steht für die hinterste Spalte. Muss ggbfalls angepasst werden, zb -2 für vorletzte Spalte
    except Exception: # falls er ihn nicht findet, den Fehler ignorieren und NA eintragen.
      topic_document_matrix_sum.iloc[i, -1] = "NA" # s.o.
    
    
### Ab hier werden die Diagnosen eingetragen  
path = os.path.join(base_path,sub_folder_data)
os.chdir(path)
diagnosen, meta = pyreadstat.read_sav('Bado_20190810_ausgefuehrt.sav') # vorbereitete Diagnosen aus SPSS werden eingelesen
diagnosen = diagnosen[["CODE","depression", "angst", "angst_depr", "keine", "andere", "PTBS"]] # berücksichtige nur die folgenden Spalten

diagnosen["patientencode"] = diagnosen["CODE"]
diagnosen = diagnosen.drop(["CODE", "session"], axis=1)
topic_document_matrix_sum["patientencode"] = topic_document_matrix_sum["session"].str[:7] 


topic_document_matrix_sum_diagnosen= pd.merge(topic_document_matrix_sum, diagnosen, on="patientencode", how="left") # übernehme diagnosen in topic document matrix sum

## Berechne die Summen der Spalten "depression", "angst", "angst_depr", "keine","andere", "PTBS". Aber addiere den nächsten Wert nur, wenn in der Spalte "patientencode" (string) ein anderer Wert ist, als in der vorherigen Zeile
### Löschen von duplizierten Zeilen mit gleichem patientencode
df_ohne_duplikate = topic_document_matrix_sum_diagnosen.drop_duplicates(subset="patientencode")

### Berechnen der Summen der Spalten "depression", "angst", "angst_depr", "keine","andere", "PTBS"
anzahl_patienten_mit_diagnosen = df_ohne_duplikate[["depression", "angst", "angst_depr", "keine", "andere", "PTBS"]].sum()
topic_document_matrix_sum_diagnosen_patient= topic_document_matrix_sum_diagnosen.drop(["patientencode"], axis=1)
    
path = os.path.join(base_path,sub_folder_data)
os.chdir(path)
anzahl_patienten_mit_diagnosen .to_excel("anzahl_patienten_mit_diagnosen .xlsx")
topic_document_matrix_sum_diagnosen_patient.to_excel("topic_document_matrix_sum_diagnosen_patient.xlsx")    
    

path = os.path.join(base_path,sub_folder_data)
os.chdir(path)
topic_document_matrix_sum.to_excel("topic_document_outcome_patient_5_250.xlsx")
