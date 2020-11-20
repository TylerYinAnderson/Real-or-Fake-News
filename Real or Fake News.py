
# coding: utf-8

# In[101]:


import numpy as np 
import pandas as pd 
import re
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from tqdm import tqdm
from hashlib import sha256
import enchant
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2
from wordcloud import WordCloud, STOPWORDS 
from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt 
import spacy
from spacy import displacy


# In[70]:


fake = pd.read_csv('Fake.csv')
true = pd.read_csv('True.csv')


# In[71]:


fake['text'][0]


# In[66]:


true.head()


# In[20]:


fake.head()


# In[21]:


def count_twitters_user(df):
    twitter_username_re = re.compile(r'@([A-Za-z0-9_]+)')
    count = 0
    list_ = []
    for text in df['text']:
        count += len(re.findall(twitter_username_re, text))
    return count


# In[22]:


twitter_users_fake_count = count_twitters_user(fake)
twitter_users_true_count = count_twitters_user(true)
fig = go.Figure()
fig.add_trace(go.Bar(x=['Fake', 'True'],
    y=[twitter_users_fake_count, twitter_users_true_count],
    name='Twitter user name Pattern',
    marker_color='indianred')
)
fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        'title': 'Unique hashtags mentions on twitter',
})
#fig = px.bar(y=[twitter_users_fake_count, twitter_users_true_count], x=['Fake', 'True'], title='Twitter user name Pattern')
fig.show()


# In[23]:


def text_size(df):
    sizes = []
    for text in tqdm(df['text']):
        len_ = len(text.split())
        sizes.append(len_)
    return np.array(sizes)

fake_size = text_size(fake)
true_size = text_size(true)
fake['len'] = fake_size
true['len'] = true_size


# In[24]:


true.head()


# In[25]:


fake['is_fake'] = 1
true['is_fake'] = 0
concat = pd.concat([fake, true])


# In[26]:


concat.head()


# In[27]:


fake_ = concat[concat['is_fake']==1]
true_ = concat[concat['is_fake']==0]
fig = go.Figure()
fig.add_trace(go.Box(y=list(fake_['len']), name='Fake',
                marker_color = 'indianred'))
fig.add_trace(go.Box(y=list(true_['len']), name = 'Real',
                marker_color = 'lightseagreen'))

fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        'title': 'Box plot',
})
fig.show()


# In[28]:


list_ = [ ]
for text in tqdm(concat['text']):
    hash_ = sha256(text.encode('utf-8')).hexdigest()
    list_.append(hash_)
concat['hash'] = list_
t = concat.groupby(['hash']).size().reset_index(name='count')
duplicate = t[t['count']>1]
print('there are ',duplicate.shape[0], 'duplicate texts')


# In[29]:


def unique_tokens(df):
    unique_tokens = set()
    for text in tqdm(df['text']):
        splited = text.split()
        for token in splited:
            unique_tokens.add(token)
    return unique_tokens

unique_tokens_fake = unique_tokens(fake)
unique_tokens_true = unique_tokens(true)


# In[30]:


twitter_users_fake_count = count_twitters_user(fake)
twitter_users_true_count = count_twitters_user(true)
fig = px.bar(y=[len(unique_tokens_fake), len(unique_tokens_true)], x=['Fake', 'True'], title='Unique tokens')
fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
})
fig.show()


# In[31]:


def check_if_exist(list_):
    d = enchant.DictWithPWL("en_US", "vocab.txt")
    count = 0
    for token in tqdm(list_):
        if not d.check(token) and not d.check(token.capitalize()):
            count+=1
    return count
count_fake = check_if_exist(unique_tokens_fake)
count_true = check_if_exist(unique_tokens_true)


# In[32]:



fig = make_subplots(rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "pie"}]])

fig.append_trace(go.Pie(values=[count_fake, len(unique_tokens_fake)-count_fake], 
                        labels=['Non exist', 'exist'], hole=.7, 
                        title='Fake News'), row=1, col=1)

fig.append_trace(go.Pie(values=[count_true, len(unique_tokens_true)-count_true], 
                        labels=['Non exist', 'exist'], hole=.7, 
                        title='Real News'), row=1, col=2)
fig.show()


# In[74]:


tqdm.pandas()
def preprocess(df):
    stopwords = nltk.corpus.stopwords.words('english')
    df['text_pre'] = df['text']
    df['text_pre'] = df['text_pre'].progress_apply(lambda x : x.lower())
    df['text_pre'] = df['text_pre'].progress_apply(lambda x : x.split(" "))
    df['text_pre'] = df['text_pre'].progress_apply(lambda x : [item for item in x if item not in stopwords])
    df['text_pre'] = df['text_pre'].progress_apply(lambda x : " ".join(x))
#    df['text_pre'] = df['text_pre'].str.replace('@[^\s]+', "")
    df['text_pre'] = df['text_pre'].str.replace('https?:\/\/.*[\r\n]*', '')
    df['text_pre'] = df['text_pre'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
    df['text_pre'] = df['text_pre'].str.replace('\d+', '')
    df['text_pre'] = df['text_pre'].str.replace('[^\w\s]', '')
    return df

fake = preprocess(fake)
true = preprocess(true)


# In[50]:


def unique_tokens2(df):
    unique_tokens = set()
    for text in tqdm(df['text_pre']):
        splited = text.split()
        for token in splited:
            unique_tokens.add(token)
    return unique_tokens

unique_tokens_fake2 = unique_tokens2(fake)
unique_tokens_true2 = unique_tokens2(true)


# In[51]:


fig = go.Figure()
fig.add_trace(go.Bar(y=[len(unique_tokens_fake2), len(unique_tokens_true2)], 
                         x=['Fake', 'True'], 
                        marker_color='lightsalmon'
))
fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
})
fig.show()


# In[75]:


count_fake = check_if_exist(unique_tokens_fake2)
count_true = check_if_exist(unique_tokens_true2)
fig = make_subplots(rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "pie"}]])

fig.append_trace(go.Pie(values=[count_fake, len(unique_tokens_fake2)-count_fake], 
                        labels=['Non exist', 'exist'], hole=.7, 
                        title='Fake News'), row=1, col=1)

fig.append_trace(go.Pie(values=[count_true, len(unique_tokens_true2)-count_true], 
                        labels=['Non exist', 'exist'], hole=.7, 
                        title='Real News'), row=1, col=2)
fig.show()


# In[78]:


def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in     vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

list_fake = get_top_n_words(fake['text_pre'], 25)
list_true = get_top_n_words(true['text_pre'], 25)


# In[79]:


new_list_words = [ seq[0] for seq in list_true ]
new_list_values = [ seq[1] for seq in list_true ]

fig = go.Figure()
fig.add_trace(go.Bar(y=new_list_values, 
                         x=new_list_words, 
                        marker_color='lightsalmon'
))
fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        'title': 'Real news Frequency words'
})
fig.show()


# In[80]:


def get_wrong_tokens(list_):
    d = enchant.DictWithPWL("en_US", "vocab.txt")
    tokens = set()
    for token in tqdm(list_):
        if not d.check(token) and not d.check(token.capitalize()):
            tokens.add(token)
    return tokens

def get_top_n_words2(corpus, n=None, vocabulary=None):
    vec = CountVectorizer(vocabulary=vocabulary).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in     vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

wrong = get_wrong_tokens(unique_tokens_true2)
wrong_true = get_top_n_words2(true['text_pre'], n=100, vocabulary=wrong)
wrong = get_wrong_tokens(unique_tokens_fake2)
wrong_fake = get_top_n_words2(fake['text_pre'], n=100, vocabulary=wrong)

new_list_words = [ seq[0] for seq in wrong_true ]
new_list_values = [ seq[1] for seq in wrong_true ]

fig = go.Figure()
fig.add_trace(go.Bar(y=new_list_values, 
                         x=new_list_words, 
                        marker_color='lightsalmon'
))
fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        'title': 'Real chi2'
})
fig.show()


# In[81]:


concat2 = pd.concat([fake, true])
concat2.head()


# In[83]:


vectorizer = CountVectorizer()
concat2 = pd.concat([fake, true])
X = vectorizer.fit_transform(concat2['text_pre'])
chi2score = chi2(X,concat2['is_fake'])[0]


# In[84]:


wscores = dict(zip(vectorizer.get_feature_names(), chi2score))


# In[85]:


dict_ = {k: v for k, v in sorted(wscores.items(), key=lambda item: item[1], reverse=True)}
keys = list(dict_.keys())
values = list(dict_.values())
fig = go.Figure()
fig.add_trace(go.Bar(y=list(values[0:50]), 
                         x=list(keys[0:50]), 
                        marker_color='lightsalmon'
))
fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
})
fig.show()


# In[87]:


def topics(model, feature_names, no_top_words):
    dict_ = {}
    for topic_idx, topic in enumerate(model.components_):
        dict_[topic_idx] = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
    return dict_
lda = LatentDirichletAllocation(random_state=42).fit(X)
topic_all = topics(lda, vectorizer.get_feature_names(), 15)


# In[88]:


vectorizer_fake = CountVectorizer()
vectorizer_true = CountVectorizer()

X_fake = vectorizer_fake.fit_transform(fake['text_pre'])
X_true = vectorizer_true.fit_transform(true['text_pre'])

lda_fake = LatentDirichletAllocation(random_state=42, n_components=5).fit(X_fake)
lda_true = LatentDirichletAllocation(random_state=42, n_components=5).fit(X_true)

topic_true = topics(lda_true, vectorizer_true.get_feature_names(), 15)
topic_fake = topics(lda_fake, vectorizer_fake.get_feature_names(), 15)


# In[92]:


def plot_clouds(dict_, title):
    for topic, words in zip(dict_.keys(), dict_.values()):
        cloud = " ".join(words)
        wordcloud = WordCloud(width = 800, height = 800, 
                        background_color ='white',  
                        min_font_size = 10).generate(cloud) 
  
        plt.figure(figsize = (4, 8), facecolor = None) 
        plt.imshow(wordcloud) 
        plt.axis("off") 
        plt.tight_layout(pad = 0) 
        plt.title(title + ' Topics '+ str(topic))
        plt.show() 


# In[93]:


plot_clouds(topic_fake, 'Fake news Topics')


# In[100]:


nlp = spacy.load("en_core_web_sm")
def get_ent(df):
    vocab = set()
    for text in tqdm(df['text']):
        doc = nlp(text)
        for ent in doc.ents:
            vocab.add(ent.text)
    return vocab


# In[102]:


vect = TfidfVectorizer()
X = vect.fit_transform(concat2['text_pre'])
y = concat2['is_fake']


# In[103]:


def select(X, y): 
    dict_ = {}
    for i in tqdm(range(1, 11)):
        value = X.shape[1] * i * 0.1
        X_new = SelectKBest(chi2, k=int(value)).fit_transform(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.33, random_state=42)
        clf = LogisticRegression()
        model = clf.fit(X_train, y_train)
        predict = model.predict(X_test)
        score = accuracy_score(y_test, predict)
        dict_[str(int(value))] = score
    return dict_

dict_ = select(X, y)


# In[104]:


fig = px.line(x=list(dict_.keys()), y=list(dict_.values()))
fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
})
fig.show()


# In[105]:


vect = TfidfVectorizer()
X = vect.fit_transform(concat2['text_pre'])
y = concat2['is_fake']

new_feature = [] 
s = SelectKBest(chi2, k=1)
X_new = s.fit_transform(X, y)
mask = s.get_support()
for bool, feature in zip(mask, vect.get_feature_names()):
    if bool:
        new_feature.append(feature)


# In[106]:


new_feature


# In[107]:


result = []
for text in concat2['text_pre']:
    if 'reuters' in text:
        result.append(0)
    else:
        result.append(1)
accuracy_score(concat2['is_fake'], result)


# In[108]:


concat2['text_pred_less_reuters'] = concat2['text_pre'].apply(lambda x : x.replace('reuters', ''))

vect = TfidfVectorizer()
X = vect.fit_transform(concat2['text_pred_less_reuters'])
y = concat2['is_fake']

new_feature = [] 
s = SelectKBest(chi2, k=1)
X_new = s.fit_transform(X, y)
mask = s.get_support()
for bool, feature in zip(mask, vect.get_feature_names()):
    if bool:
        new_feature.append(feature)


# In[110]:


new_feature


# In[109]:


result = []
for text in concat2['text_pre']:
    if 'said' in text:
        result.append(0)
    else:
        result.append(1)


# In[111]:


accuracy_score(concat2['is_fake'], result)

