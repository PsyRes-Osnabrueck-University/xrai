from HanTa import HanoverTagger as ht
import re
import os
import json
# Create path and sub folders
base_path = "C:/Users/JLU-SU/JLUbox/Transkriptanalysen (Christopher Lalk)/2 TOPIC MODELING/Analysen/" # base_path = "C:/Users/Christopher/JLUbox/Transkriptanalysen/2 TOPIC MODELING/Analysen/"
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
sys.path.insert(0,r'C:\Users\Christopher\PycharmProjects\Bertopic\topictuner')
from topictuner import TopicModelTuner as TMT

from sentence_transformers import SentenceTransformer
import onnxruntime


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


#BERTopic model
model = BERTopic(language="German", vectorizer_model = cv, top_n_words = 10, nr_topics=150, n_gram_range = (1,3), calculate_probabilities=True, verbose = True)
topics, probabilities = model.fit_transform(sentences, embeddings=embeddings)


# Topics speichern
path = os.path.join(base_path,sub_folder_data)
os.chdir(path)
df_topics_ther_10_150 = model.get_topic_info()
df_topics_ther_10_150.to_excel("topics_ther_10_150.xlsx")

# BERTopic speichern
path = os.path.join(base_path,sub_folder_data)
os.chdir(path)
model.save("Ther_10words_150")


# BERTopic Model ther_10words_200
#################################################################################################################################
# Embeddings sind fertig!


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

path = os.path.join(base_path,sub_folder_data)
os.chdir(path)
model = BERTopic.load("Pat_10words_150")



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
session_list = [i[:-4] for i in topic_document_matrix_sum.index]
session_list = [i.replace(" (fertig)", "") for i in session_list]
session_list = [i.replace("(fertig)", "") for i in session_list]
session_list = [i.replace("u15", "") for i in session_list]
topic_document_matrix_sum.insert(0, column="session", value=session_list)

## lese SPSS Daten ein
path = os.path.join(base_path,sub_folder_processing)
os.chdir(path)
sitzungsbogen = pd.read_spss('Stundenbögen_20190810.sav')
sitzungsbogen["session"] = sitzungsbogen["CODE"] + "_" + sitzungsbogen["INSTANCE"]

for i in len(topic_document_matrix_sum):
    topic_document_matrix_sum


topic_document_matrix_sum.at[df_fin["File"].tolist().index(sfn), 'Patient'] = transcript_clean
