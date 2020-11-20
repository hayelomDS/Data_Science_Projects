#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import sklearn
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
import pyprind
import os
import urllib.request
import urllib.parse
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv("Tweets.csv")
print("The current shape of dataset is:",  df.shape)
print("\nThe current columns are: \n", df.columns)


# In[3]:


tweets = df[['airline','text','airline_sentiment']]
print("The shape of the dataset is:", tweets.shape)
tweets.head()


# In[4]:


# info of our features and labels 
print(tweets.info())
print(tweets.describe(include='all'))


# In[5]:


# check the counts of the different airlines 
print(tweets["airline"].value_counts())

# ckeck if we have any null values 
print(tweets.isnull().sum())


# In[6]:


# saving the reduced data for future use with 3 cols
tweets.to_csv('airlines_data.csv', index=False, encoding='utf-8')


# In[7]:


def clean_txt(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text) # remove @ sign 
    text = re.sub(r'#', '', text) # remove '#'
    text = re.sub(r'https?:\/\/\S+', '', text) # remove hyper link 
    text = re.sub(r'[^a-zA-Z#]+',' ', text) 
    text = re.sub('[\W]+', ' ', text.lower())
    text = ' '.join([word for word in text.split()
                    if len(word) > 2])
    tokenized = ' '.join([word for word in text.split() 
                    if word not in stop])
    document  = ' '.join([porter.stem(word) 
                    for word in tokenized.split()]) 
    return document


# In[8]:


import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

stop = stopwords.words('english')
porter = PorterStemmer()

# check to make sure the function works
clean_txt(tweets.loc[2, 'text'])


# In[9]:


tweets['text'] = tweets['text'].apply(clean_txt)


# In[10]:


tweets.head()


# ##  Story telling and Vizualization
# 
# - What are the most common words in the entire dataset?
# - what are the most common words in the dataset for the negative and positive tweets, respectively. 

# In[11]:


from wordcloud import WordCloud

# first convert all words into one list 
all_words = ' '.join([word for word in tweets['text']])
#all_words


# In[12]:


wordcloud = WordCloud(width=800, height=500, random_state=21, 
                      max_font_size=110).generate(all_words)
plt.figure(figsize= (10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()


# In[13]:


positive_words = ' '.join([word for word in tweets['text'][tweets['airline_sentiment']
                                                            == 'positive']])

wordcloud = WordCloud(width=800, height=500, random_state=21, 
                      max_font_size=110).generate(positive_words)
plt.figure(figsize= (10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()


# In[14]:


negative_words = ' '.join([word for word in tweets['text'][tweets['airline_sentiment']
                                                            == 'negative']])

wordcloud = WordCloud(width=800, height=500, random_state=21, 
                      max_font_size=110).generate(negative_words)
plt.figure(figsize= (10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()


# In[15]:


neutral_words = ' '.join([word for word in tweets['text'][tweets['airline_sentiment']
                                                            == 'neutral']])

wordcloud = WordCloud(width=800, height=500, random_state=21, 
                      max_font_size=110).generate(neutral_words)
plt.figure(figsize= (10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()


# In[16]:


tweet_two_class = tweets.drop(tweets.loc[tweets['airline_sentiment']=='neutral'].index)
tweets_all_class = tweets


# In[17]:


from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV


# In[18]:


tweets_all_class.head()
tweet_two_class.head()


# ### All classes

# In[19]:


X_all = tweets_all_class.loc[:, 'text']
y_all = tweets_all_class.loc[:, 'airline_sentiment']


# In[20]:


label_class_all = {label: index for index, label in 
               enumerate(np.unique(tweets_all_class['airline_sentiment']))}

print(label_class_all)


# In[21]:


y_all_transform = y_all.map(label_class_all)
y_all_transform


# In[22]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all_transform, 
                                    train_size=0.80,test_size=0.20, 
                                    random_state=101, stratify = y_all_transform)


# In[23]:


print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape) 


# ### With 2 classification positive and negative only

# In[24]:


X_two = tweet_two_class.loc[:, 'text']
y_two = tweet_two_class.loc[:, 'airline_sentiment']


# In[25]:


label_class_two = {label: index for index, label in 
               enumerate(np.unique(tweet_two_class['airline_sentiment']))}

print(label_class_two)


# In[26]:


y_transform_two = y_two.map(label_class_two)
y_transform_two


# In[27]:


X_train_two, X_test_two, y_train_two, y_test_two = train_test_split(X_two, y_transform_two, 
                                    train_size=0.80,test_size=0.20, 
                                    random_state=101, stratify = y_transform_two)


# In[28]:


print(X_train_two.shape, X_test_two.shape)
print(y_train_two.shape, y_test_two.shape) 


# # Model 1

# ## Logistic Regression 
# 
# - Grig search using 5 fold and 10 fold cross validation.
# - Hyperparameter tuning to optimize model 
# - able to train and test on the same data
# - Not enough data to split for train, valid and test

# In[29]:


from nltk.stem.porter import PorterStemmer

porter = PorterStemmer()

def tokenizer(text):
    return text.split()

def tokenizer_porter(text):
    return [porter.stem(word) for word in text]

tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=False,
                        preprocessor=None)

param_grid = [{'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 5.0, 10.0, 100.0]},
              {'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'vect__use_idf':[False],
               'vect__norm':[None],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 5.0, 10.0, 100.0]},
              ]


lr_tfidf = Pipeline([('vect', tfidf),
                     ('clf', LogisticRegression(random_state=0, solver='liblinear'))])


gs_lr_five_fold = GridSearchCV(lr_tfidf, param_grid,
                           scoring='accuracy',
                           cv=5,
                           verbose=2,
                           n_jobs=-1)

gs_lr_ten_fold = GridSearchCV(lr_tfidf, param_grid,
                           scoring='accuracy',
                           cv=10,
                           verbose=2,
                           n_jobs=-1)


# # With 3 class

# ### With 5 fold CV

# In[51]:


gs_lr_five_fold.fit(X_train, y_train)


# In[52]:


print(gs_lr_five_fold.best_params_)
gs_lr_five_fold.best_score_


# In[53]:


optim_clf = gs_lr_five_fold.best_estimator_
print('Test Accuracy: %.3f' % optim_clf.score(X_test, y_test))


# ### with 10 fold CV

# In[54]:


gs_lr_ten_fold.fit(X_train, y_train)


# In[55]:


print(gs_lr_ten_fold.best_params_)
gs_lr_ten_fold.best_score_


# In[56]:


optim_clf = gs_lr_ten_fold.best_estimator_
print('Test Accuracy: %.3f' % optim_clf.score(X_test, y_test))


# ## With 2 class 5 fold CV

# In[57]:


gs_lr_five_fold.fit(X_train_two, y_train_two)


# In[58]:


print(gs_lr_five_fold.best_params_)
gs_lr_five_fold.best_score_


# In[60]:


optim_clf = gs_lr_five_fold.best_estimator_
print('Test Accuracy: %.3f' % optim_clf.score(X_test_two, y_test_two))


# ## With 2 class 10 fold CV

# In[61]:


gs_lr_ten_fold.fit(X_train_two, y_train_two)


# In[62]:


print(gs_lr_ten_fold.best_params_)
gs_lr_ten_fold.best_score_


# In[63]:


optim_clf = gs_lr_ten_fold.best_estimator_
print('Test Accuracy: %.3f' % optim_clf.score(X_test_two, y_test_two))


# In[64]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report


# In[66]:


print(classification_report(y_test_two, optim_clf.predict(X_test_two), digits=4))


# # Model 2 Naive Bayes

# In[67]:


nb = MultinomialNB()
lr = LogisticRegression()
tf = TfidfVectorizer()


# In[68]:


tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=False,
                        preprocessor=None)


param_grid = [{'vect__ngram_range': [(1, 1), (1, 2)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter]},
              {'vect__ngram_range': [(1, 1), (1, 2)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'vect__use_idf':[False],
               'vect__norm':[None]},
              ]


nb_tfidf = Pipeline([('vect', tfidf),
                     ('clf', MultinomialNB(alpha=1.0))])


gs_nb_tfidf_five = GridSearchCV(nb_tfidf, param_grid,
                           scoring='accuracy',
                           cv=5,
                           verbose=2,
                           n_jobs=-1)

gs_nb_tfidf_ten = GridSearchCV(nb_tfidf, param_grid,
                           scoring='accuracy',
                           cv=10,
                           verbose=2,
                           n_jobs=-1)


# ## With 3 classes 5 fold CV

# In[69]:


gs_nb_tfidf_five.fit(X_train, y_train)


# In[71]:


print(gs_nb_tfidf_five.best_params_)
gs_nb_tfidf_five.best_score_


# In[72]:


optim_clf = gs_nb_tfidf_five.best_estimator_
print('Test Accuracy: %.3f' % optim_clf.score(X_test, y_test))


# In[73]:


print(classification_report(y_test, optim_clf.predict(X_test), digits=4))


# ## With 3 classes 10 fold CV

# In[74]:


gs_nb_tfidf_ten.fit(X_train, y_train)


# In[79]:


print(gs_nb_tfidf_ten.best_params_)
gs_nb_tfidf_ten.best_score_


# In[76]:


optim_clf = gs_nb_tfidf_ten.best_estimator_
print('Test Accuracy: %.3f' % optim_clf.score(X_test, y_test))


# In[77]:


print(classification_report(y_test, optim_clf.predict(X_test), digits=4))


# ## With 2 class 5 fold CV

# In[78]:


gs_nb_tfidf_five.fit(X_train_two, y_train_two)


# In[80]:


print(gs_nb_tfidf_five.best_params_)
gs_nb_tfidf_five.best_score_


# In[81]:


optim_clf = gs_nb_tfidf_five.best_estimator_
print('Test Accuracy: %.3f' % optim_clf.score(X_test_two, y_test_two))


# In[82]:


print(classification_report(y_test_two, optim_clf.predict(X_test_two), digits=4))


# ## With 2 class 10 fold CV

# In[85]:


gs_nb_tfidf_ten.fit(X_train_two, y_train_two)


# In[86]:


print(gs_nb_tfidf_ten.best_params_)
gs_nb_tfidf_ten.best_score_


# In[87]:


optim_clf = gs_nb_tfidf_ten.best_estimator_
print('Test Accuracy: %.3f' % optim_clf.score(X_test_two, y_test_two))


# In[88]:


print(classification_report(y_test_two, optim_clf.predict(X_test_two), digits=4))


# # Model 3 

# In[95]:


from sklearn import svm


# In[97]:


sv = svm.SVC()


# In[98]:


sv.get_params()


# In[103]:


lr.get_params()


# In[118]:


param_grid = [{'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'clf__C': [1.0, 5.0, 10.0], 
                'clf__gamma': [1, 0.1, 0.01, 0.001]}]


sv_tfidf = Pipeline([('vect', tfidf),
                     ('clf', svm.SVC(kernel='linear'))])


gs_sv_tfidf = GridSearchCV(sv_tfidf, param_grid,
                           scoring='accuracy',
                           cv=10,
                           verbose=2,
                           n_jobs=-1)


# # 3 Class

# In[113]:


gs_sv_tfidf.fit(X_train, y_train)


# In[114]:


print(gs_sv_tfidf.best_params_)
gs_sv_tfidf.best_score_


# In[116]:


optim_clf = gs_sv_tfidf.best_estimator_
print('Test Accuracy: %.3f' % gs_sv_tfidf.score(X_test, y_test))


# In[117]:


print(classification_report(y_test, optim_clf.predict(X_test), digits=4))


# In[119]:


gs_sv_tfidf.fit(X_train_two, y_train_two)


# In[120]:


optim_clf = gs_sv_tfidf.best_estimator_
print('Test Accuracy: %.3f' % gs_sv_tfidf.score(X_test_two, y_test_two))


# In[121]:


print(gs_sv_tfidf.best_params_)
gs_sv_tfidf.best_score_


# In[ ]:




