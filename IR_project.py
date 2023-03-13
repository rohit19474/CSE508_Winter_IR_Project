#!/usr/bin/env python
# coding: utf-8

# In[2]:


import re
import string
import nltk
from rouge import Rouge
from transformers import T5Tokenizer, T5ForConditionalGeneration
import pandas as pd
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer


# In[3]:


# Load the data
df = pd.read_csv('news_summary.csv', encoding='ISO-8859-1')
df=df[0:100]


# In[4]:


# Load the tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')


# In[5]:


nlp = spacy.load('en_core_web_sm')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Remove HTML tags and URLs
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'http\S+', '', text)
    
    # Tokenize the text using NLTK
    tokens = nltk.word_tokenize(text.lower())

    # Remove stopwords and punctuation
    tokens = [token for token in tokens if token not in stop_words and token not in string.punctuation]

    # Lemmatize the tokens using spaCy
    lemmas = [token.lemma_ for token in nlp(" ".join(tokens))]

    # Remove any remaining non-alphabetic tokens
    lemmas = [lemma for lemma in lemmas if lemma.isalpha()]

    # Join the lemmas back into a string
    text = " ".join(lemmas)

    return text

# Preprocess the text in the DataFrame
df['preprocessed_text'] = df['text'].apply(preprocess_text)


# In[6]:


# Extract features using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['preprocessed_text'])


# In[7]:


# Generate summaries and titles for each article
summaries = []
titles = []
for i in range(len(df)):
    # Tokenize the article text
    text = df['text'][i]
    inputs = tokenizer.encode("summarize: " + text, return_tensors='pt', max_length=512, truncation=True)

    # Generate the summary
    summary_ids = model.generate(inputs,
                                 num_beams=4,
                                 no_repeat_ngram_size=2,
                                 min_length=30,
                                 max_length=100,
                                 early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    summaries.append(summary)

    # Generate the title from the summary
    title_inputs = tokenizer.encode("summarize: " + summary, return_tensors='pt', max_length=512, truncation=True)
    title_ids = model.generate(title_inputs,
                               num_beams=4,
                               no_repeat_ngram_size=2,
                               min_length=10,
                               max_length=40,
                               early_stopping=True)
    title = tokenizer.decode(title_ids[0], skip_special_tokens=True)
    titles.append(title)


# In[8]:


# Add the summaries and titles to the DataFrame
df['generated_summary'] = summaries
df['generated_title'] = titles


# In[9]:


# Initialize the ROUGE metric
rouge = Rouge()


# In[10]:


# Calculate ROUGE scores for summaries and titles
rouge_scores_summary = []
rouge_scores_title = []
for i in range(len(df)):
    reference = df['headlines'][i]
    summary = df['generated_summary'][i]
    title = df['generated_title'][i]

    scores_summary = rouge.get_scores(summary, reference)
    rouge_scores_summary.append(scores_summary[0])

    scores_title = rouge.get_scores(title, reference)
    rouge_scores_title.append(scores_title[0])
df['rouge_scores_summary'] = rouge_scores_summary
df['rouge_scores_title'] = rouge_scores_title


# In[14]:


df = df.drop(['read_more','ctext','date'], axis=1)
df.head()


# In[32]:


df['text'][1]


# In[33]:


df['generated_summary'][1]


# In[34]:


df['generated_title'][1]


# In[44]:


df['rouge_scores_summary'][1]


# In[43]:


df['rouge_scores_title'][1]


# In[ ]:





# In[42]:


df['text'][57]


# In[40]:


df['generated_summary'][57]


# In[41]:


df['generated_title'][57]


# In[45]:


df['rouge_scores_summary'][57]


# In[46]:


df['rouge_scores_title'][57]


# In[ ]:




