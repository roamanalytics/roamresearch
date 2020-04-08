
# coding: utf-8

# ## Parse and tokenize using SciSpacy

# In[1]:


import os
import sys
import string
import pandas as pd
import numpy as np
import nltk
import gensim
import seaborn as sns
import matplotlib.pyplot as plt
from pprint import pprint
from pathlib import Path


# ## Configurations and Paths

# In[2]:


ROOT = '..'


# In[3]:


datafile_date = '2020-03-27-v5'


# In[4]:


# Configurations
basedir = ROOT + f'/data/interim/{datafile_date}/'
specify_doc_type = 1
doc_type = ['smalldocs', 'abstracts'][specify_doc_type]
DEBUG_LIMIT = None # 100  # None
SPACY_FLAG = True
EXPORT_TOKENS = True  # Recommended only for abstracts, unless topic models for full text
EXPORT_EMBEDDINGS = True


# In[5]:


out_debug_flag = f'-DEBUG_{DEBUG_LIMIT}' if DEBUG_LIMIT else ''


# In[6]:


datafile = f'{basedir}{datafile_date}-covid19-combined-{doc_type}.jsonl'
embeddir = f'{basedir}embeddings/'
embedding_file_template = f'{embeddir}{datafile_date}-covid19-combined-{doc_type}-embedding' + '-{}' + out_debug_flag + '.npy'
tokens_file_template = f'{basedir}{datafile_date}-covid19-combined-{doc_type}-tokens' + '-{}' + out_debug_flag + '.jsonl'


# In[7]:


if doc_type == 'smalldocs':
    text_column_name = 'text'
    process_cols = [text_column_name]
elif doc_type == 'abstracts':
    text_column_name = 'abstract_clean'
    process_cols = [text_column_name]
elif doc_type == 'only-abstracts':
    text_column_name = 'abstract_clean'
    process_cols = [text_column_name]


# In[8]:


json_args = {'orient': 'records', 'lines': True}
out_json_args = {'date_format': 'iso', **json_args}


# In[9]:


if not os.path.exists(datafile):
    print(datafile + ' does not exist')
    sys.exit()


# In[10]:


out_path_mode = 0o777
os.makedirs(embeddir, mode = out_path_mode, exist_ok = True)


# ## Read in text and create corpus

# In[11]:


original_df = pd.read_json(datafile, **json_args)


# In[12]:


documents = original_df[text_column_name]


# In[13]:


len(documents)


# ## SciSpacy parsing

# In[14]:


df = original_df.copy()


# In[15]:


if DEBUG_LIMIT:  # False: 
    df = df.loc[:DEBUG_LIMIT, :].copy()
DEBUG_LIMIT


# ### Prep

# In[16]:


import spacy
import scispacy


# In[17]:


if SPACY_FLAG:
    model_name = "en_core_sci_lg"
    model = spacy.load(model_name)


# In[18]:


embedding_outfile = {}
tokens_outfile = {}
for name in process_cols:
    embedding_outfile[name] = embedding_file_template.format('spacy-' + model_name + '-' + name)
    tokens_outfile[name] = tokens_file_template.format('spacy-' + model_name)


# In[19]:


def spacy_embedding(text, model, debug_identifier=None, empty_value=None):
    if not text:
        return empty_value
    try:
        doc_vector = model(text).vector
        return doc_vector
    except:
        if debug_identifier:
            print('Spacy Embedding error with debug_id=' + str(debug_identifier))
        raise


# In[20]:


def spacy_embedding_ents_tokens(text, model, debug_identifier=None, empty_value=None):
    if not text:
        return (empty_value, empty_value, empty_value)
    try:
        sdoc = model(text)
        sdoc_vector = sdoc.vector
        sdoc_ents = list(ent.text for ent in sdoc.ents)
        sdoc_tokens = list(tok.text for tok in sdoc)
        return (sdoc_vector, sdoc_ents, sdoc_tokens)
    except:
        if debug_identifier:
            print('Spacy Embedding error with debug_id=' + str(debug_identifier))
        raise


# ### Get Spacy Embeddings

# In[21]:


if SPACY_FLAG and not EXPORT_TOKENS and text_column_name in process_cols:
    doc_embeddings_text = np.array([spacy_embedding(value, model, empty_value=[])
                                      for index, value in df[text_column_name].items()])
    doc_embeddings_text.shape


# In[22]:


if SPACY_FLAG and EXPORT_TOKENS and text_column_name in process_cols:
    doc_embeddings_ents_tokens_text = np.array([spacy_embedding_ents_tokens(value, model, empty_value=[])
                                      for index, value in df[text_column_name].items()])
    
    doc_embeddings_text, doc_ents_text, doc_tokens_text = zip(*doc_embeddings_ents_tokens_text)
    doc_embeddings_text = np.array(doc_embeddings_text)
    print(doc_embeddings_text.shape)


# In[23]:


# Check embedding length
set([len(x) for x in doc_embeddings_text])


# ### Write

# In[24]:


if SPACY_FLAG and EXPORT_EMBEDDINGS and text_column_name in process_cols:
    np.save(embedding_outfile[text_column_name], doc_embeddings_text)
    print(embedding_outfile[text_column_name])


# In[25]:


tokens_column_name = 'abstract_tokens_scispacy'
ents_column_name = 'abstract_ent_scispacy'


# In[26]:


if SPACY_FLAG and EXPORT_TOKENS and text_column_name in process_cols:
    new_df = df.copy()
    new_df[tokens_column_name] = pd.Series(doc_tokens_text)
    new_df[ents_column_name] = pd.Series(doc_ents_text)
    new_df.to_json(tokens_outfile[text_column_name], **out_json_args)
    print(tokens_outfile[text_column_name])

