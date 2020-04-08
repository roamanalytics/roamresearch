
# coding: utf-8

# # Read datafiles, merge, and lightly clean

# In[1]:


import os
import json
import datetime
from pprint import pprint
from copy import deepcopy

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm


# #### Configuration

# In[2]:


ROOT = '..'


# In[3]:


datafile_date = '2020-03-27-v5'


# In[4]:


PROCESS_SMALL_DOCS = False  # Small docs are the individual paragraphs in the text


# In[5]:


SOURCE_FILES = {
    'COMM-USE': ROOT + f'/data/raw/{datafile_date}/comm_use_subset/',
    'BioRxiv': ROOT + f'/data/raw/{datafile_date}/biorxiv_medrxiv/',
    'NONCOMM': ROOT + f'/data/raw/{datafile_date}/noncomm_use_subset/',
    'PMC': ROOT + f'/data/raw/{datafile_date}/custom_license/',
}


# In[6]:


metadata_file = ROOT + f'/data/raw/{datafile_date}/metadata.csv'


# In[7]:


outdir = ROOT + f'/data/interim/{datafile_date}/'
outfile = f'{outdir}{datafile_date}-covid19-combined.jsonl'
outfile_small_docs = f'{outdir}{datafile_date}-covid19-combined-smalldocs.jsonl'
outfile_abstracts = f'{outdir}{datafile_date}-covid19-combined-abstracts.jsonl'
json_args = {'orient': 'records', 'lines': True}
out_json_args = {'date_format': 'iso', **json_args}


# In[8]:


out_path_mode = 0o777
os.makedirs(outdir, mode = out_path_mode, exist_ok = True)


# ## Helper Functions

# Some functions taken and modified from https://www.kaggle.com/xhlulu/cord-19-eda-parse-json-and-generate-clean-csv

# In[9]:


def format_name(author):
    middle_name = " ".join(author['middle'])
    
    if author['middle']:
        return " ".join([author['first'], middle_name, author['last']])
    else:
        return " ".join([author['first'], author['last']])


def format_affiliation(affiliation):
    text = []
    location = affiliation.get('location')
    if location:
        text.extend(list(affiliation['location'].values()))
    
    institution = affiliation.get('institution')
    if institution:
        text = [institution] + text
    return ", ".join(text)

def format_authors(authors, with_affiliation=False):
    name_ls = []
    
    for author in authors:
        name = format_name(author)
        if with_affiliation:
            affiliation = format_affiliation(author['affiliation'])
            if affiliation:
                name_ls.append(f"{name} ({affiliation})")
            else:
                name_ls.append(name)
        else:
            name_ls.append(name)
    
    return ", ".join(name_ls)

def format_body(body_text):
    texts = [(di['section'], di['text']) for di in body_text]
    texts_di = {di['section']: "" for di in body_text}
    
    for section, text in texts:
        texts_di[section] += text

    body = ""

    for section, text in texts_di.items():
        body += section
        body += "\n\n"
        body += text
        body += "\n\n"
    
    return body

def format_bib(bibs):
    if type(bibs) == dict:
        bibs = list(bibs.values())
    bibs = deepcopy(bibs)
    formatted = []
    
    for bib in bibs:
        bib['authors'] = format_authors(
            bib['authors'], 
            with_affiliation=False
        )
        doi = None
        pmid = None
        other_ids = bib.get('other_ids')
        if other_ids:
            doi = other_ids.get('DOI')
            pmid = other_ids.get('PMID')
        formatted_ls = [str(bib[k]) for k in ['title', 'authors', 'venue', 'year']]
        if doi:
            formatted_ls.extend(doi)
        if pmid:
            formatted_ls.extend(['PMID' + p for p in pmid])
        formatted.append(", ".join(formatted_ls))

    return "\n ".join(formatted)


# In[10]:


def bib_titles(bibs):
    result = {}
    for key, bib in bibs.items():
        result[key] = bib['title']
    return result

def extract_small_docs(main_doc_id, body_text, bib_titles_dict):
    result = []
    for i, di in enumerate(body_text):
        ref_titles = []
        for ref in di['cite_spans']:
            title = bib_titles_dict.get(ref['ref_id'])
            if title:
                ref_titles.append(title)
        result.append((main_doc_id, i, di['text'], di['section'], ref_titles))
    return result


# In[11]:


def load_files(dirname):
    filenames = os.listdir(dirname)
    raw_files = []

    for filename in tqdm(filenames):
        filename = dirname + filename
        file = json.load(open(filename, 'rb'))
        raw_files.append(file)
    
    return raw_files

def generate_clean_df(all_files, prepare_small_docs=False):
    cleaned_files = []
    small_docs = []
    
    for file in tqdm(all_files):
        if prepare_small_docs:
            bib_titles_dict = bib_titles(file['bib_entries'])
            docs = extract_small_docs(file['paper_id'], file['body_text'], bib_titles_dict)
        else:
            docs = []

        features = [
            file['paper_id'],
            file['metadata']['title'],
            format_authors(file['metadata']['authors']),
            format_authors(file['metadata']['authors'], 
                           with_affiliation=True),
            format_body(file['abstract']),
            format_body(file['body_text']),
            format_bib(file['bib_entries']),
            file['metadata']['authors'],
            file['bib_entries'],
            len(docs)
        ]

        cleaned_files.append(features)
        if prepare_small_docs:
            small_docs.extend(docs)

    col_names = ['paper_id', 'title', 'authors',
                 'affiliations', 'abstract', 'text', 
                 'bibliography','raw_authors','raw_bibliography',
                'num_small_docs']

    clean_df = pd.DataFrame(cleaned_files, columns=col_names)
    clean_df.head()
    
    if prepare_small_docs:
        small_docs_df = pd.DataFrame(small_docs, 
                                     columns=['paper_id', 'small_doc_num', 'text', 'section', 'ref_titles'])
        return clean_df, small_docs_df
    else:
        return clean_df


# ## Load Data

# ### Load Metadata

# In[12]:


metadata_df = pd.read_csv(metadata_file)


# In[13]:


metadata_df.head()


# ### Clean Metadata

# In[14]:


metadata_df['publish_year'] = metadata_df['publish_time'].astype(str).apply(lambda d: 
                                                                d[:4] if d[0] in ('1', '2') else
                                                                '19xx' if d == 'nan' else
                                                                # d[2:6] if d.startswith("'[") else
                                                                '')


# In[15]:


metadata_df['publish_year'].unique()


# ### Load Data Files

# In[16]:


dfd = {}
small_docs = {}
for name, indir in SOURCE_FILES.items():
    print(f'Loading {name} from {indir}')
    data_files = load_files(indir)
    print(f"Cleaning {name} {len(data_files)} files" )
    if PROCESS_SMALL_DOCS:
        dfd[name], small_docs[name] = generate_clean_df(data_files, prepare_small_docs=True)
    else:
        dfd[name] = generate_clean_df(data_files)


# In[17]:


dfd['COMM-USE'].head()


# ### Combine data from text files

# In[18]:


for name, df in dfd.items():
    df['dataset'] = name


# In[19]:


df_combined = pd.concat(dfd.values(), ignore_index=True, sort=False)


# In[20]:


df_combined.head()


# In[21]:


if PROCESS_SMALL_DOCS:
    for name, df in small_docs.items():
        df['dataset'] = name
    df_combined_small_docs = pd.concat(small_docs.values(), ignore_index=True, sort=False)
    print(df_combined_small_docs.shape)


# In[22]:


if PROCESS_SMALL_DOCS:
    print(df_combined_small_docs.columns)


# ### Join Metadata and Data Files

# In[23]:


df = metadata_df.copy()


# In[24]:


df_joined = df.join(df_combined.set_index('paper_id'), how='left', on='sha', rsuffix='_ft')


# In[25]:


df_joined.head()


# In[26]:


df_joined_ft = df_joined[~ df_joined['sha'].isnull()].copy()


# In[27]:


df_joined_ft.shape


# ### Clean abstract

# In[28]:


df_joined_ft['abstract_clean'] = df_joined_ft['abstract'].fillna('')


# In[29]:


df_joined_ft['abstract_clean'] = df_joined_ft['abstract_clean'].apply(lambda x: x[9:] if x.lower().startswith('abstract') else x)


# ### Create citation ref

# In[30]:


df_joined_ft['cite_ad'] = df_joined_ft['authors'].fillna('').str.split(',').str[0].str.split(' ').str.join('_') + '_' + df_joined_ft['publish_year']


# ### Write data

# In[31]:


df_joined_ft.columns


# In[32]:


# Warning: This file is over 2GB
df_joined_ft.to_json(outfile, **out_json_args)
outfile


# In[33]:


if PROCESS_SMALL_DOCS:
    df_combined_small_docs.to_json(outfile_small_docs, **out_json_args)


# In[34]:


df_joined_ft.head()


# In[35]:


df_joined_ft.loc[:, ['cord_uid', 'sha', 'abstract_clean',
                     'cite_ad', 'title', 'authors', 'publish_year', 'publish_time', 'dataset',
                                 'pmcid', 'pubmed_id', 'doi'
                    ]].to_json(outfile_abstracts, **out_json_args)

