
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import gensim
import nltk
from nltk.corpus import stopwords
from gensim import corpora
from gensim import models
from pprint import pprint
from gensim.models import LsiModel
from gensim.models import LdaModel, LdaMulticore
from gensim.models.coherencemodel import CoherenceModel
from nltk.stem.wordnet import WordNetLemmatizer
from wordcloud import WordCloud
csv.field_size_limit(1000000000)

data = pd.read_csv('state-of-the-union.csv')

np_data = data.to_numpy()
data = pd.DataFrame(np_data, columns=['year','speech'])

with open('stopwords-en.csv', mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    stop_words=[]
    for row in csv_reader:
        stop_words.append(row['a'])

stopwords = set(stopwords.words('english')) 

lemmatizer = WordNetLemmatizer()
data.speech = data.speech.str.replace('[\n]',' ')
data.speech = data.speech.str.replace('[^\w\s]',' ')
data.speech = data.speech.str.replace('[^A-Za-z]',' ')
data.speech = data.speech.str.replace('  ',' ')
for i in range(data.speech.size):
    data.speech[i] = nltk.word_tokenize(data.speech[i])
    filtered_tokens = [lemmatizer.lemmatize(token) for token in data.speech[i] if token not in stopwords]
    data.speech[i] = ' '.join(filtered_tokens)

def create_wordcloud(df):
    long_string = ','.join(df)
    wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=5, contour_color='steelblue')
    wordcloud.generate(long_string)
    return wordcloud.to_image()

cloud = create_wordcloud(data.speech)
plt.imshow(cloud)
plt.title('Word Cloud for entire dataset')

cloud = create_wordcloud(data.speech[0:10])
plt.imshow(cloud)
plt.title('Word Cloud for oldest 10 speech in the dataset')

cloud = create_wordcloud(data.speech[215:225])
plt.imshow(cloud)
plt.title('Word Cloud for latest 10 speech in the  dataset')

texts = [[word for word in document.lower().split() if word not in stop_words]
         for document in data.speech]
from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

# Only keep words that appear more than once
texts = [[token for token in text if frequency[token] > 1] for text in texts]

dictionary = corpora.Dictionary(texts)
print(dictionary)

bow_corpus = [dictionary.doc2bow(text) for text in texts]

word_counts = [[(dictionary[id], count) for id, count in line] for line in bow_corpus]


tfidf = models.TfidfModel(bow_corpus)
doc = tfidf[bow_corpus[2]]

corpus_tfidf = tfidf[bow_corpus]

lsi_model = LsiModel(corpus_tfidf, id2word=dictionary, num_topics=250, decay=0.5)  # initialize an LSI transformation
corpus_lsi = lsi_model[corpus_tfidf] 
topics = lsi_model.print_topics(5)
for topic in topics:
    print(topic)

def compute_coherence_values(dictionary, doc_term_matrix, doc_clean, stop, start=2, step=3):
    coherence_values = []
    model_list = []
    for num_topics in range(start, stop, step):
        # generate LSA model
        model = LsiModel(doc_term_matrix, num_topics=num_topics, id2word = dictionary)  # train model
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=doc_clean, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values

model_list, coherence_values = compute_coherence_values(dictionary, corpus_tfidf, texts, 1000)

x = range(2, 1000, 3)
plt.plot(x, coherence_values)
plt.xlabel("Number of Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()

def compute_coherence_values_LDA(dictionary, doc_term_matrix, doc_clean, stop, start=2, step=5):
    coherence_values = []
    model_list = []
    for num_topics in range(start, stop, step):
        model = LdaModel(doc_term_matrix, num_topics=num_topics, id2word = dictionary)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=doc_clean, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values


x = range(2, 300, 5)
plt.plot(x, coherence_values)
plt.xlabel("Number of Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()


lda_model = LdaMulticore(corpus=bow_corpus,
                         id2word=dictionary,
                         random_state=100,
                         num_topics=200,
                         passes=50,
                         chunksize=1000,
                         batch=False,
                         alpha='asymmetric',
                         decay=0.5,
                         offset=64,
                         eta=None,
                         eval_every=0,
                         iterations=100,
                         gamma_threshold=0.001,
                         per_word_topics=True)


corpus_lda = lda_model[bow_corpus]

topics = lda_model.print_topics(5,num_words=10)
for topic in topics:
    print(topic)

speech=''
decade = []
for i in range(data.shape[0]):
    year = int(data['year'][i])
    speech = speech + " " + data['speech'][i]
    if (year+1)%10==0:
        decade.append(speech)
        speech=''
decade.append(speech)


len(decade)

def make_tfidf(data):
    texts = [[word for word in data.lower().split() if word not in stop_words]]
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1

    # Only keep words that appear more than once
    texts = [[token for token in text if frequency[token] > 1] for text in texts]
    dictionary = corpora.Dictionary(texts)
    bow_corpus = [dictionary.doc2bow(text) for text in texts]
#     tfidf = models.TfidfModel(bow_corpus)
#     corpus_tfidf = tfidf[bow_corpus]
    return dictionary, bow_corpus 

def LDA(corpus, dictionary):
    lda_model = LdaMulticore(corpus=corpus,
                             id2word=dictionary,
                             random_state=100,
                             num_topics=250,
                             passes=10,
                             chunksize=1000,
                             batch=False,
                             alpha='asymmetric',
                             decay=0.5,
                             offset=64,
                             eta=None,
                             eval_every=0,
                             iterations=100,
                             gamma_threshold=0.001,
                             per_word_topics=True)
    return lda_model.print_topics(5,num_words=10)

dictionary, bow_corpus = make_tfidf(decade[1])

dictionary, bow_corpus = make_tfidf(decade[0])
topics = LDA(bow_corpus, dictionary)
print(dictionary)
print('Topics for this decade are')
print(topics)

dictionary, bow_corpus = make_tfidf(decade[1])
topics = LDA(bow_corpus, dictionary)
print(dictionary)
print('Topics for this decade are')
print(topics)

dictionary, bow_corpus = make_tfidf(decade[2])
topics = LDA(bow_corpus, dictionary)
print(dictionary)
print('Topics for this decade are')
print(topics)

dictionary, bow_corpus = make_tfidf(decade[3])
topics = LDA(bow_corpus, dictionary)
print(dictionary)
print('Topics for this decade are')
print(topics)
dictionary, bow_corpus = make_tfidf(decade[4])
topics = LDA(bow_corpus, dictionary)
print(dictionary)
print('Topics for this decade are')
print(topics)

dictionary, bow_corpus = make_tfidf(decade[5])
topics = LDA(bow_corpus, dictionary)
print(dictionary)
print('Topics for this decade are')
print(topics)
dictionary, bow_corpus = make_tfidf(decade[6])
topics = LDA(bow_corpus, dictionary)
print(dictionary)
print('Topics for this decade are')
print(topics)

dictionary, bow_corpus = make_tfidf(decade[7])
topics = LDA(bow_corpus, dictionary)
print(dictionary)
print('Topics for this decade are')
print(topics)
dictionary, bow_corpus = make_tfidf(decade[8])
topics = LDA(bow_corpus, dictionary)
print(dictionary)
print('Topics for this decade are')
print(topics)
dictionary, bow_corpus = make_tfidf(decade[9])
topics = LDA(bow_corpus, dictionary)
print(dictionary)
print('Topics for this decade are')
print(topics)

dictionary, bow_corpus = make_tfidf(decade[10])
topics = LDA(bow_corpus, dictionary)
print(dictionary)
print('Topics for this decade are')
print(topics)

dictionary, bow_corpus = make_tfidf(decade[12])
topics = LDA(bow_corpus, dictionary)
print(dictionary)
print('Topics for this decade are')
print(topics)
dictionary, bow_corpus = make_tfidf(decade[13])
topics = LDA(bow_corpus, dictionary)
print(dictionary)
print('Topics for this decade are')
print(topics)
dictionary, bow_corpus = make_tfidf(decade[14])
topics = LDA(bow_corpus, dictionary)
print(dictionary)
print('Topics for this decade are')
print(topics)
dictionary, bow_corpus = make_tfidf(decade[15])
topics = LDA(bow_corpus, dictionary)
print(dictionary)
print('Topics for this decade are')
print(topics)
dictionary, bow_corpus = make_tfidf(decade[16])
topics = LDA(bow_corpus, dictionary)
print(dictionary)
print('Topics for this decade are')
print(topics)
dictionary, bow_corpus = make_tfidf(decade[17])
topics = LDA(bow_corpus, dictionary)
print(dictionary)
print('Topics for this decade are')
print(topics)
dictionary, bow_corpus = make_tfidf(decade[18])
topics = LDA(bow_corpus, dictionary)
print(dictionary)
print('Topics for this decade are')
print(topics)
dictionary, bow_corpus = make_tfidf(decade[19])
topics = LDA(bow_corpus, dictionary)
print(dictionary)
print('Topics for this decade are')
print(topics)
dictionary, bow_corpus = make_tfidf(decade[20])
topics = LDA(bow_corpus, dictionary)
print(dictionary)
print('Topics for this decade are')
print(topics)
dictionary, bow_corpus = make_tfidf(decade[21])
topics = LDA(bow_corpus, dictionary)
print(dictionary)
print('Topics for this decade are')
print(topics)
dictionary, bow_corpus = make_tfidf(decade[22])
topics = LDA(bow_corpus, dictionary)
print(dictionary)
print('Topics for this decade are')
print(topics)
AP = pd.read_csv('ap.csv')

np_AP = AP.to_numpy()
AP = pd.DataFrame(np_AP, columns=['ID','story'])
AP = AP.dropna()

np_AP = AP.to_numpy()
AP = pd.DataFrame(np_AP, columns=['ID','story'])
AP = AP.dropna()

lemmatizer = WordNetLemmatizer()
AP.story = AP.story.str.replace('[\n]',' ')
# AP.story = AP.story.str.replace('[^\w\s]',' ')
AP.story = AP.story.str.replace('[^A-Za-z]',' ')
AP.story = AP.story.str.replace('  ',' ')
for i in range(AP.story.size):
    AP.story[i] = nltk.word_tokenize(AP.story[i])
    filtered_tokens = [lemmatizer.lemmatize(token) for token in AP.story[i]]
    AP.story[i] = ' '.join(filtered_tokens)

cloud = create_wordcloud(AP.story)
plt.imshow(cloud)
plt.title('Word Cloud for AP dataset')

textsAP = [[word for word in document.lower().split() if word not in stopwords]
         for document in AP.story]

from collections import defaultdict
frequency = defaultdict(int)
for text in textsAP:
    for token in text:
        frequency[token] += 1
textsAP = [[token for token in text if frequency[token] > 1] for text in textsAP]

dictionaryAP = corpora.Dictionary(textsAP)

corpusAP = [dictionaryAP.doc2bow(text) for text in textsAP]

model_list_nq, coherence_values_nq = compute_coherence_values_LDA(dictionaryAP, corpusAP, textsAP, 300)

x = range(2, 300, 5)
plt.plot(x, coherence_values_nq)
plt.xlabel("Number of Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()

lda_model3 = LdaMulticore(corpus=corpusAP,
                         id2word=dictionaryAP,
                         random_state=22,
                         num_topics=100,
                         passes=10,
                         chunksize=1000,
                         batch=False,
                         alpha='asymmetric',
                         decay=0.5,
                         offset=64,
                         eta=None,
                         eval_every=0,
                         iterations=100,
                         gamma_threshold=0.001,
                         per_word_topics=True)
lda_model3.print_topics(5,num_words=20)
