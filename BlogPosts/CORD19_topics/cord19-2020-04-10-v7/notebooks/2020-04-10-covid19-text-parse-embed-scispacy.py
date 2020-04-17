
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


datafile_date = '2020-04-10-v7'


# In[4]:


# Configurations
basedir = ROOT + f'/data/interim/{datafile_date}/'
specify_doc_type = 1
doc_type = ['smalldocs', 'abstracts'][specify_doc_type]
DEBUG_LIMIT = None # 100  # None
SPACY_FLAG = True
EXPORT_TOKENS = True  # Recommended only for abstracts, unless topic models for full text
EXPORT_EMBEDDINGS = True
CALCULATE_SIMILARITIES = 5 


# In[5]:


out_debug_flag = f'-DEBUG_{DEBUG_LIMIT}' if DEBUG_LIMIT else ''


# In[6]:


# input files
datafile = f'{basedir}{datafile_date}-covid19-combined-{doc_type}.jsonl'
specter_file = ROOT + f'/data/raw/{datafile_date}/' + 'cord19_specter_embeddings_2020-04-10.csv'
# output files
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


# In[11]:


if CALCULATE_SIMILARITIES and CALCULATE_SIMILARITIES > 0:
    gs_index_tempfile = f'{basedir}tmp/gs_index'
    os.makedirs(f'{basedir}tmp', mode = out_path_mode, exist_ok = True)


# ## Read in text and create corpus

# In[12]:


original_df = pd.read_json(datafile, **json_args)


# In[13]:


documents = original_df[text_column_name]


# In[14]:


len(documents)


# ## SciSpacy parsing

# In[15]:


df = original_df.copy()


# In[16]:


if DEBUG_LIMIT:  # False: 
    df = df.loc[:DEBUG_LIMIT, :].copy()
DEBUG_LIMIT


# ### Prep

# In[17]:


import spacy
import scispacy


# In[18]:


if SPACY_FLAG:
    model_name = "en_core_sci_lg"
    model = spacy.load(model_name)


# In[19]:


embedding_outfile = {}
tokens_outfile = {}
for name in process_cols:
    embedding_outfile[name] = embedding_file_template.format('spacy-' + model_name + '-' + name)
    tokens_outfile[name] = tokens_file_template.format('spacy-' + model_name)


# In[20]:


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


# In[21]:


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

# In[22]:


if SPACY_FLAG and not EXPORT_TOKENS and text_column_name in process_cols:
    doc_embeddings_text = np.array([spacy_embedding(value, model, empty_value=[])
                                      for index, value in df[text_column_name].items()])
    doc_embeddings_text.shape


# In[23]:


if SPACY_FLAG and EXPORT_TOKENS and text_column_name in process_cols:
    doc_embeddings_ents_tokens_text = np.array([spacy_embedding_ents_tokens(value, model, empty_value=[])
                                      for index, value in df[text_column_name].items()])
    
    doc_embeddings_text, doc_ents_text, doc_tokens_text = zip(*doc_embeddings_ents_tokens_text)
    doc_embeddings_text = np.array(doc_embeddings_text)
    print(doc_embeddings_text.shape)


# In[24]:


# Check embedding length
set([len(x) for x in doc_embeddings_text])


# ### Calculate Similarities

# In[25]:


from gensim.similarities.docsim import Similarity


# In[26]:


if CALCULATE_SIMILARITIES and CALCULATE_SIMILARITIES > 0:
    num_best = CALCULATE_SIMILARITIES + 1
    if SPACY_FLAG:
        print('Calculating scispacy similarities index')
        spacy_index = Similarity(gs_index_tempfile, doc_embeddings_text, num_features=200, num_best=num_best)
    print('Reading specter embeddings')
    specter_df = pd.read_csv(specter_file, header=None, index_col=0)
    print('Calculating specter similarities index')
    specter_index = Similarity(gs_index_tempfile, specter_df.to_numpy(),
                               num_features=specter_df.shape[1], num_best=num_best)


# In[27]:


if CALCULATE_SIMILARITIES and CALCULATE_SIMILARITIES > 0:
    if SPACY_FLAG:
        print('Calculating scispacy similarities')    
        df['sims_scispacy_idx'] = [[id_val[0] for id_val in sims[:CALCULATE_SIMILARITIES] if id_val[0] != i]
                                for i, sims in enumerate(spacy_index)]
        df['sims_scispacy_cord_uid'] = df['sims_scispacy_idx'].apply(lambda lst: [df.loc[i, 'cord_uid'] for i in lst])
    print('Calculating specter similarities')
    sims_specter_cord_uid_s = pd.Series([[specter_df.index[id_val[0]] for id_val in sims[:CALCULATE_SIMILARITIES]
                                     if id_val[0] != i] 
                              for i, sims in enumerate(specter_index)],
                                    index=specter_df.index,
                                   name='sims_specter_cord_uid')
    print('Joining specter similarities')
    sims_specter_cord_uid_s = sims_specter_cord_uid_s[~ sims_specter_cord_uid_s.index.duplicated(keep='last')]
    df = df.join(sims_specter_cord_uid_s, on='cord_uid')


# ### Write

# In[28]:


if SPACY_FLAG and EXPORT_EMBEDDINGS and text_column_name in process_cols:
    np.save(embedding_outfile[text_column_name], doc_embeddings_text)
    print(embedding_outfile[text_column_name])


# In[29]:


tokens_column_name = 'abstract_tokens_scispacy'
ents_column_name = 'abstract_ent_scispacy'


# In[30]:


if SPACY_FLAG and EXPORT_TOKENS and text_column_name in process_cols:
    new_df = df.copy()
    new_df[tokens_column_name] = pd.Series(doc_tokens_text)
    new_df[ents_column_name] = pd.Series(doc_ents_text)
    new_df.to_json(tokens_outfile[text_column_name], **out_json_args)
    print(tokens_outfile[text_column_name])

