#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('pip', 'install xgboost')


# In[3]:


import numpy as np
import pandas as pd
import re
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier


# In[4]:


df = pd.read_csv('final_hateXplain.csv')


# In[5]:


df.head()


# In[6]:


df.isnull()


# In[7]:


df.duplicated()


# In[8]:


encoder = LabelEncoder()
df['label'] = encoder.fit_transform(df['label'])


# In[9]:


X = df['comment'] + ' ' + df['Race'] + ' ' + df['Religion'] + ' ' + df['Gender'] + ' ' + df['Sexual Orientation'] + ' ' + df['Miscellaneous']



# In[10]:


wn = WordNetLemmatizer()
def preprocessing(content):
  for i in range(0, len(content)):
      content = re.sub('[^a-zA-Z]', ' ', content)
      content = content.lower()
      content = [wn.lemmatize(word) for word in content.split() if not word in stopwords.words('english')]
      content = ' '.join(content)
      return content
X = X.apply(preprocessing)


# In[11]:


X.tail()


# In[12]:


cv = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
x_res = cv.fit_transform(X).toarray()
y = df['label']


# In[13]:


cv.get_params()


# In[14]:


cv.get_feature_names_out()[:50]


# In[15]:


X_train, X_test, Y_train, Y_test = train_test_split(x_res, y, test_size=0.2, random_state=42)


# In[16]:


count_df = pd.DataFrame(X_train, columns=cv.get_feature_names_out())
count_df.head()


# In[17]:


from xgboost import XGBClassifier


# In[18]:


models = {
     'LogisticRegression': LogisticRegression(max_iter=1000),
     'MultinomialNB': MultinomialNB(),
     'RandomForestClassifier' : RandomForestClassifier(random_state=42),
     'PassiveAggressiveClassifier': PassiveAggressiveClassifier(),
     'XGClassifier': XGBClassifier()
}

for name, model in models.items():
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    accuracy = round(accuracy_score(Y_test, Y_pred)*100,2)
    print(name, 'Analysis\n')
    print(f'Accuracy: {accuracy}%\n')


# In[19]:


X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[28]:


pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=1000))
])

param_grid = {
    'tfidf__max_features': [10000, 15000, 20000],
    'tfidf__ngram_range': [(1,1), (1, 2), (1, 3)],
    'clf__C': [0.1, 0.5, 1, 5, 10],
    'clf__penalty': ['l2']
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5)
grid_search.fit(X_train, Y_train)


# In[37]:


print(f"Best parameters: {grid_search.best_params_}")
y_pred = grid_search.best_estimator_.predict(X_test)
accuracy = round(accuracy_score(Y_test, Y_pred), 4)*100
conf_matrix = confusion_matrix(Y_test, Y_pred)
classification_rep = classification_report(Y_test, Y_pred)
print(f"\nAccuracy: {accuracy}%")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{classification_rep}")


# In[ ]:




