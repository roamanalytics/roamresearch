
# coding: utf-8

# # Topics Modeling using Mallet (through gensim wrapper)

# ## Initialization

# ### Preliminaries & Configurations

# In[1]:


import os
import sys
import string
import numpy as np
import datetime
import pandas as pd
import json
import re
import nltk
import gensim
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


#!pip install pyLDAvis
#!pip install panel
import pyLDAvis
import pyLDAvis.gensim
import panel as pn


# In[ ]:


pn.extension()  # This can cause Save to error "Requested Entity to large"; Clear this cell's output after running
None


# In[4]:


MALLET_ROOT = '/home/jovyan'


# In[5]:


mallet_home = os.path.join(MALLET_ROOT, 'mallet-2.0.8')
mallet_path = os.path.join(mallet_home, 'bin', 'mallet')
mallet_stoplist_path = os.path.join(mallet_home, 'stoplists', 'en.txt')


# In[6]:


ROOT = '..'


# In[7]:


# Configurations
datafile_date = '2020-03-27-v5'
basedir = ROOT + f'/data/interim/{datafile_date}/'
# parser = 'moana'
parser = 'scispacy'
parser_model = 'spacy-en_core_sci_lg'
# Inputs
datafile = f'{basedir}{datafile_date}-covid19-combined-abstracts-tokens-{parser_model}.jsonl'
text_column_name = 'abstract_clean'
tokens_column_name = f'abstract_tokens_{parser}'
ent_column_name = f'abstract_ent_{parser}'
json_args = {'orient': 'records', 'lines': True}


# In[8]:


# Outputs
outdir = ROOT + f'/results/{datafile_date}/'
model_out_dir = ROOT + f'/models/topics-abstracts-{datafile_date}-{parser}/'
model_path = model_out_dir + 'mallet_models/'
gs_model_path = model_path + 'gs_models/'
gs_model_path_prefix = gs_model_path + f'{datafile_date}-covid19-combined-abstracts-'
out_json_args = {'date_format': 'iso', **json_args}
web_out_dir = outdir + f'topics-abstracts-{datafile_date}-{parser}-html/'


# In[9]:


if not os.path.exists(datafile):
    print(datafile + ' does not exist')
    sys.exit()


# In[10]:


out_path_mode = 0o777
os.makedirs(model_out_dir, mode = out_path_mode, exist_ok = True)
os.makedirs(model_path, mode = out_path_mode, exist_ok = True)
os.makedirs(gs_model_path, mode = out_path_mode, exist_ok = True)
os.makedirs(outdir, mode = out_path_mode, exist_ok = True)
os.makedirs(web_out_dir, mode = out_path_mode, exist_ok = True)


# In[11]:


import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.WARNING)

logging.getLogger("gensim").setLevel(logging.WARNING)


# In[12]:


with open(mallet_stoplist_path, 'r') as fp:
    stopwords = set(fp.read().split())
len(stopwords)


# In[13]:


stopwords.update([
    'doi', 'preprint', 'copyright', 'peer', 'reviewed', 'org', 'https', 'et', 'al', 'author', 'figure', 
    'rights', 'reserved', 'permission', 'used', 'using', 'biorxiv', 'fig', 'fig.', 'al.',
    'di', 'la', 'il', 'del', 'le', 'della', 'dei', 'delle', 'una', 'da',  'dell',  'non', 'si'
])    # from https://www.kaggle.com/danielwolffram/topic-modeling-finding-related-articles
len(stopwords)


# ### Read in text and create corpus

# In[14]:


original_df = pd.read_json(datafile, **json_args)


# In[15]:


documents = original_df[text_column_name]
orig_tokens = original_df[tokens_column_name]
if ent_column_name in original_df.columns:
    ents = original_df['abstract_ent_scispacy'].apply(lambda lst: ['_'.join(k.lower().split()) for k in lst if len(k.split()) > 1])
else:
    ents = None


# In[16]:


len(documents)


# In[17]:


punctuation = string.punctuation + "”“–"  # remove both slanted double-quotes
# leave '#$%*+-/<=>'
nonnumeric_punctuation = r'!"&()\,.:;?@[]^_`{|}~' + "'" + "'""”“–’" + ' '

def normalize_token(token):
    if token in nonnumeric_punctuation:
        return None
    if token in stopwords:
        return None
    if token == token.upper():
        return token
    return token.lower()

def normalize_token_list(tokens):
    result = []
    for tok in tokens:
        ntok = normalize_token(tok)
        if ntok:
            result.append(ntok)
    return result


# In[18]:


texts = orig_tokens.apply(normalize_token_list)


# In[ ]:


dictionary = gensim.corpora.Dictionary(texts)


# In[ ]:


corpus = [dictionary.doc2bow(text) for text in texts]


# In[ ]:


sorted(dictionary.values())[:5]


# ## Topic model collections -- vary corpus and n

# ### Prepare corpus collections (various options)

# In[ ]:


corpora = {}
# corpora['text'] = corpus


# ##### Filter scispacy ents

# In[ ]:


from collections import Counter
if ents is not None:
    ents_counter = Counter()
    for x in ents.iteritems():
        for w in x[1]:
            ents_counter[w] += 1
    ents_common = [k for k, c in ents_counter.items() if c >= 5]
    len(ents_common)


# ##### Extended token sets 

# In[ ]:


dictionary = gensim.corpora.Dictionary(texts)
if ents is not None:
    dictionary.add_documents([ents_common])


# In[ ]:


# Several combinations attempted, but 'text-ents' was most useful
if ents is not None:
    corpora['text-ents'] = (texts + ents).apply(dictionary.doc2bow)


# In[ ]:


corpora.keys()


# ### HTML Templates

# In[ ]:


html_template = '''
<!DOCTYPE html>
<html>
<meta charset="UTF-8">
<head>
  <title>{0}</title>
{1}
</head>
<body>
<h2>{0}</h2>
{2}
</body>
</html>
'''

html_style = '''
<style>
table {
  font-family: "Trebuchet MS", Arial, Helvetica, sans-serif;
  border-collapse: collapse;
  width: 100%;
}

td, th {
  border: 1px solid #ddd;
  padding: 8px;
}

tr:nth-child(even){background-color: #f2f2f2;}

tr:hover {background-color: #ddd;}

th {
  padding-top: 12px;
  padding-bottom: 12px;
  text-align: left;
  background-color: #0099FF;
  color: white;
}
</style>
'''

html_style_cols = '''
<style>
table {
  font-family: "Trebuchet MS", Arial, Helvetica, sans-serif;
  border-collapse: collapse;
  width: 100%;
}

td, th {
  border: 1px solid #ddd;
  padding: 8px;
}

td:nth-child(even){background-color: #f2f2f2;}

td:hover {background-color: #ddd;}

th {
  padding-top: 12px;
  padding-bottom: 12px;
  text-align: left;
  background-color: #0099FF;
  color: white;
}
</style>
'''


# ### Build models

# In[ ]:


num_topics = [60]  # number of topics


# In[ ]:


cmallet = {}
for c in corpora.keys():
    cmallet[c] = {}
    for i in num_topics:
        print('Building model for %s (%s topic)' % (c,i))
        prefix = os.path.join(model_path, c, str(i), '')
        os.makedirs(prefix, mode = out_path_mode, exist_ok = True)
        cmallet[c][i] = gensim.models.wrappers.ldamallet.LdaMallet(mallet_path, corpora[c], id2word=dictionary, optimize_interval=10,
                                                            prefix=prefix, 
                                                                   num_topics=i, iterations='2500 --random-seed 42')


# ### Plot

# In[ ]:


vis_data = {}
gensim_lda_model = {}
for c in cmallet.keys():
    vis_data[c] = {}
    gensim_lda_model[c] = {}
    for i in cmallet[c].keys():
        gensim_lda_model[c][i] = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(cmallet[c][i])
        vis_data[c][i] = pyLDAvis.gensim.prepare(gensim_lda_model[c][i], corpora[c], 
                                                   dictionary=cmallet[c][i].id2word, mds='tsne')
        pyLDAvis.save_html(vis_data[c][i], outdir + f'pyldavis_{c}_{i}.html')


# #### Save Gensim Mallet Models

# In[ ]:


for c in gensim_lda_model.keys():
    for i in gensim_lda_model[c].keys():
        gensim_lda_model[c][i].save(f'{gs_model_path_prefix}gensim-mallet-model_{c}_{i}.pickle4', 
                                    separately=[], sep_limit=134217728, pickle_protocol=4)
        print(f'{gs_model_path_prefix}gensim-mallet-model_{c}_{i}.pickle4')


# #### Save _Relevant_ terms for topics (from pyLDAviz)

# In[ ]:


num_terms = 50


# In[ ]:


def sorted_terms(data, topic=1, rlambda=1, num_terms=30):
    """Returns a dataframe using lambda to calculate term relevance of a given topic."""
    tdf = pd.DataFrame(data.topic_info[data.topic_info.Category == 'Topic' + str(topic)])
    if rlambda < 0 or rlambda > 1:
        rlambda = 1
    stdf = tdf.assign(relevance=rlambda * tdf['logprob'] + (1 - rlambda) * tdf['loglift'])
    rdf = stdf[['Term', 'relevance']]
    if num_terms:
        return rdf.sort_values('relevance', ascending=False).head(num_terms).set_index(['Term'])
    else:
        return rdf.sort_values('relevance', ascending=False).set_index(['Term'])


# In[ ]:


topic_lists = {}
for corp, cdict in vis_data.items():
    for numtops in cdict.keys():
        model_topic_lists_dict = {}
        for topnum in range(numtops):
            s = sorted_terms(vis_data[corp][numtops], topnum + 1, rlambda=.5, num_terms=num_terms)
            terms = s.index
            model_topic_lists_dict['Topic ' + str(topnum + 1)] = np.pad(terms, (0, num_terms - len(terms)),
                                                                               'constant', constant_values='')
        topic_lists[corp + '-' + str(numtops)] = pd.DataFrame(model_topic_lists_dict)


# In[ ]:


topic_lists.keys()


# In[ ]:


# !pip install openpyxl


# In[ ]:


# Save relevant topics - write to xlsx (one corp-numtopics per sheet)
with pd.ExcelWriter(outdir + f'topics-relevant-words-abstracts-{datafile_date}-{num_terms}terms.xlsx') as writer: 
    for sheetname, dataframe in topic_lists.items():
        dataframe.to_excel(writer, sheet_name=sheetname)
print(outdir + f'topics-relevant-words-abstracts-{datafile_date}-{num_terms}terms.xlsx')


# #### Save Relevant Topics as html

# In[ ]:


# Save relevant topics - write to html
out_topics_html_dir = web_out_dir
for corp_numtopics, dataframe in topic_lists.items():
    os.makedirs(out_topics_html_dir + corp_numtopics, mode = out_path_mode, exist_ok = True)
    ofname = out_topics_html_dir + corp_numtopics + '/' + 'relevant_terms.html'
    with open(ofname, 'w') as ofp:
        column_tags = [f'<a href="Topic_{i+1:02d}.html" target="_blank">{name}</a>' 
                       for i, name in enumerate(dataframe.columns)]
        temp_df = dataframe.copy()
        temp_df.columns = column_tags
        temp_df = temp_df.applymap(lambda x: ' '.join(x.split('_')))
        temp_df = temp_df.set_index(np.arange(1, len(temp_df) + 1))
        html_table = temp_df.to_html(escape=False)
        html_str = html_template.format('Most Relevant Terms per Topic', html_style_cols, html_table)
        ofp.write(html_str)
    print(ofname)


# In[ ]:


# topic_lists['text-ents-60']


# ### Create dataframes of topic model collections

# In[ ]:


ctopicwords_df = {}
for c in cmallet.keys():
    ctopicwords_df[c] = {}
    for i in cmallet[c].keys():
        ctopicwords_df[c][i] = pd.read_table(cmallet[c][i].ftopickeys(), header=None, names=['id', 'weight', 'wordlist'])


# In[ ]:


REMOVED = []
def normalize_topic_words(words):
    results = []
    for w in words:
        if w in nonnumeric_punctuation:
            pass
        elif w[-1] == 's' and w[:-1] in words:
            # remove plural
            REMOVED.append(w)
        elif w != w.lower() and w.lower() in words:
            # remove capitalized
            REMOVED.append(w)
        else:
            results.append(w)
    return results


# In[ ]:


# Clean words
for c in ctopicwords_df.keys():
    for i in ctopicwords_df[c].keys():
        ctopicwords_df[c][i]['wordlist'] = ctopicwords_df[c][i]['wordlist'].apply(lambda x: ' '.join(normalize_topic_words(x.split())))


# In[ ]:


# set(REMOVED)


# In[ ]:


for c in ctopicwords_df.keys():
    for i in ctopicwords_df[c].keys():
        ctopicwords_df[c][i].drop(['id'], axis=1, inplace=True)
        ctopicwords_df[c][i]['topwords'] = ctopicwords_df[c][i].wordlist.apply(lambda x: ' '.join(x.split()[:3]))
        ctopicwords_df[c][i]['topten'] = ctopicwords_df[c][i].wordlist.apply(lambda x: ' '.join(x.split()[:10]))
        if True:  # use pyLDAvis order
            rank_order_new_old = vis_data[c][i].to_dict()['topic.order']
            rank_order_old_new = [None] * len(rank_order_new_old)
            for new, old in enumerate(rank_order_new_old):
                rank_order_old_new[old - 1] = new
            ctopicwords_df[c][i]['rank'] = np.array(rank_order_old_new) + 1
        else:
            ctopicwords_df[c][i]['rank'] = ctopicwords_df[c][i].weight.rank(ascending=False)
        ctopicwords_df[c][i]['topicnum'] = ctopicwords_df[c][i].apply(lambda row: ('t%02d' % row['rank']), axis=1)
        ctopicwords_df[c][i]['label'] = ctopicwords_df[c][i].apply(lambda row: row['topicnum'] + ' ' + row['topwords'], axis=1)


# In[ ]:


# doctopics
cdoctopics_df = {}
for c in cmallet.keys():
    cdoctopics_df[c] = {}
    for n in cmallet[c].keys():
        cdoctopics_df[c][n] = pd.read_table(cmallet[c][n].fdoctopics(), header=None, names=['id']+[i for i in range(n)])
        cdoctopics_df[c][n].drop(['id'], axis=1, inplace=True)
# cdoctopics_df[c][n].head()


# In[ ]:


# Reorder topics
for c in cdoctopics_df.keys():
    for n in cdoctopics_df[c].keys():
# (include top 3 topics in name)        cdoctopics_df[c][n] = cdoctopics_df[c][n].T.join(ctopicwords_df[c][n][['rank', 'label']]).set_index('label').sort_values('rank').drop(['rank'], axis=1).T
        cdoctopics_df[c][n] = cdoctopics_df[c][n].T.join(ctopicwords_df[c][n][['rank', 'topicnum']]).set_index('topicnum').sort_values('rank').drop(['rank'], axis=1).T
        cdoctopics_df[c][n].T.index.rename('topic', inplace=True)
cdoctopics_df[c][n].head()


# ### Save documents

# In[ ]:


# Save topicwords
for c in ctopicwords_df.keys():
    for i in ctopicwords_df[c].keys():
        ctopicwords_df[c][i].sort_values('rank').to_csv(outdir + 'topickeys_sorted_%s_%d.txt' % (c, i), index_label='original_order')
        print(outdir + 'topickeys_sorted_%s_%d.txt' % (c, i))
        # ctopicwords_df[c][i].sort_values('rank').to_excel('out/topickeys_sorted_%s_%d.xlsx' % (c, i), index_label='original_order')


# In[ ]:


# Save doctopics
for c in cdoctopics_df.keys():
    for n in cdoctopics_df[c].keys():
        cdoctopics_df[c][n].to_csv(outdir + 'doctopic_%s_%d.csv' % (c, n), index_label='original_order')
        print(outdir + 'doctopic_%s_%d.csv' % (c, n))


# In[ ]:


match_covid19_regex = re.compile('covid-19|sars-cov-2|2019-ncov|sars coronavirus 2|2019 novel coronavirus',
                                re.IGNORECASE)
def match_covid19(text):
    return bool(match_covid19_regex.match(text))


# In[ ]:


# Prepare to save docs by topics
predominant_doc_dfd = {}
predominant_doc_df = original_df[['cite_ad', 'title', 'authors', 'publish_year', 'publish_time', 'dataset',
                                 'pmcid', 'pubmed_id', 'doi', 'sha', 'abstract_clean']].copy()
predominant_doc_df['mentions_COVID-19'] = predominant_doc_df['abstract_clean'].apply(match_covid19)
predominant_doc_df['publish_time'] = predominant_doc_df['publish_time'].dt.strftime('%Y-%m-%d')
for c in cdoctopics_df.keys():
    predominant_doc_dfd[c] = {}
    for n in cdoctopics_df[c].keys():
        predominant_doc_dfd[c][n] = {}
        predominant_doc_df['major_topics'] = cdoctopics_df[c][n].apply(lambda r: {f't{i + 1:02d}': val for i, val in enumerate(r) if val >= 0.3}, axis=1)
        for i, topic_name in enumerate(cdoctopics_df[c][n].columns):        
            temp_df = predominant_doc_df[(predominant_doc_df['major_topics'].apply(lambda x: topic_name in x))].copy()
            temp_df['topic_weight'] = temp_df.major_topics.apply(lambda x: x.get(topic_name))
            temp_df = temp_df.sort_values(['topic_weight'], axis=0, ascending=False)
            predominant_doc_dfd[c][n][i] = temp_df


# In[ ]:


# Save docs by topics - write to json and tsv
for c in predominant_doc_dfd.keys():
    for n in predominant_doc_dfd[c].keys():
        outfile_central_docs_base = outdir + f'topics-central-docs-abstracts-{datafile_date}-{c}-{n}'
        temp_dfs = []
        for i, dataframe in predominant_doc_dfd[c][n].items():
            temp_df = dataframe[['title', 'authors', 'publish_year', 'publish_time', 'dataset', 'sha', 'abstract_clean']].reset_index()
            temp_df['Topic'] = i + 1
            temp_dfs.append(temp_df)
        result_df = pd.concat(temp_dfs)
        print(outfile_central_docs_base + '.{jsonl, txt}')
        result_df.to_json(outfile_central_docs_base + '.jsonl', **out_json_args)
        result_df.to_csv(outfile_central_docs_base + '.txt', sep='\t')


# In[ ]:


# Save docs by topics - write to excel
for c in predominant_doc_dfd.keys():
    for n in predominant_doc_dfd[c].keys():
        print(outdir + f'topics-central-docs-abstracts-{datafile_date}-{c}-{n}.xlsx')
        with pd.ExcelWriter(outdir + f'topics-central-docs-abstracts-{datafile_date}-{c}-{n}.xlsx') as writer: 
            for i in predominant_doc_dfd[c][n].keys():
                sheetname = f'Topic {i+1}'
                predominant_doc_dfd[c][n][i].drop(columns=['abstract_clean', 'cite_ad', 'major_topics']).to_excel(writer, sheet_name=sheetname)


# In[ ]:


# Modify dataframe for html
for c in predominant_doc_dfd.keys():
    for n in predominant_doc_dfd[c].keys():
        for i in predominant_doc_dfd[c][n].keys():
            predominant_doc_dfd[c][n][i]['pmcid'] = predominant_doc_dfd[c][n][i]['pmcid'].apply(lambda xid: f'<a href="https://www.ncbi.nlm.nih.gov/pmc/articles/{xid}" target="_blank">{xid}</a>' if not pd.isnull(xid) else '')
            predominant_doc_dfd[c][n][i]['pubmed_id'] = predominant_doc_dfd[c][n][i]['pubmed_id'].apply(lambda xid: f'<a href="https://www.ncbi.nlm.nih.gov/pubmed/{xid}" target="_blank">{xid}</a>' if not pd.isnull(xid) else '')
            predominant_doc_dfd[c][n][i]['doi'] = predominant_doc_dfd[c][n][i]['doi'].apply(lambda xid: f'<a href="https://doi.org/{xid}" target="_blank">{xid}</a>' if not pd.isnull(xid) else '')
            predominant_doc_dfd[c][n][i].columns = [' '.join(c.split('_')) for c in predominant_doc_dfd[c][n][i].columns]


# In[ ]:


# Save doc by topics - write to html
out_topics_html_dir = outdir + f'topics-central-docs-abstracts-{datafile_date}-html/'
os.makedirs(out_topics_html_dir, mode = out_path_mode, exist_ok = True)
for c in predominant_doc_dfd.keys():
    for n in predominant_doc_dfd[c].keys():
        ofdir = out_topics_html_dir + f'{c}-{n}/'
        os.makedirs(ofdir, mode = out_path_mode, exist_ok = True)   
        print(ofdir)
        for i in predominant_doc_dfd[c][n].keys():
            ofname = ofdir + f'Topic_{i+1:02d}.html'
            with open(ofname, 'w') as ofp:
                html_table = (predominant_doc_dfd[c][n][i]
                                .drop(columns=['cite ad', 'sha', 'major topics', 'abstract clean'])
                                .copy()
                                .set_index(np.arange(1, len(predominant_doc_dfd[c][n][i])+1))
                                .to_html(escape=False))
                html_str = html_template.format(f'Topic {i+1:02d}', html_style, html_table)
                ofp.write(html_str)

