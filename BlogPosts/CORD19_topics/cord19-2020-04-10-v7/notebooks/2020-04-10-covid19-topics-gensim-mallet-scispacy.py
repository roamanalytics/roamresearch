
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
#!pip install pycld2
import pyLDAvis
import pyLDAvis.gensim
import panel as pn
import pycld2 as cld2
# !pip install openpyxl


# In[ ]:


pn.extension()  # This can cause Save to error "Requested Entity to large"; Clear this cell's output after running
None


# In[4]:


MALLET_ROOT = '/home/jovyan'


# In[5]:


mallet_home = os.path.join(MALLET_ROOT, 'mark/Systems/mallet-2.0.8')
mallet_path = os.path.join(mallet_home, 'bin', 'mallet')
mallet_stoplist_path = os.path.join(mallet_home, 'stoplists', 'en.txt')


# In[6]:


ROOT = '..'


# In[7]:


# Configurations
datafile_date = '2020-04-10-v7'
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
# Other configurations
MODIFIED_LDAVIS_URL = 'https://cdn.jsdelivr.net/gh/roamanalytics/roamresearch@master/BlogPosts/CORD19_topics/ldavis.v1.0.0-roam.js'
random_seed = 42
model_build_workers = 4


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
if 'keyterms' in original_df.columns:
    # keyterms = original_df['keyterms'].apply(lambda x: [k.lower() for k in x])
    keyterms = original_df['keyterms'].apply(lambda lst: ['_'.join(k.lower().split()) for k in lst])
else:
    keyterms = None
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


nonnumeric_punctuation


# In[19]:


texts = orig_tokens.apply(normalize_token_list)


# In[20]:


dictionary = gensim.corpora.Dictionary(texts)


# In[21]:


corpus = [dictionary.doc2bow(text) for text in texts]


# In[22]:


sorted(dictionary.values())[:5]


# ## Topic model collections -- vary corpus and n

# ### Prepare corpus collections (various options)

# In[23]:


corpora = {}
# corpora['text'] = corpus


# ##### Filter by language

# In[24]:


def predict_language(text):
    try:
        isReliable, _, details = cld2.detect(text, isPlainText=True)
    except cld2.error:
        return ('ERROR', 0, '')
    if isReliable:
        lang_prob = details[0][2]
        if lang_prob > 70:
            return (details[0][1], lang_prob, text)
        elif lang_prob == 0:
            return ('', 0, '')
        # abstract likely in two languages
        _, _, details, vectors = cld2.detect(text, isPlainText=True,
                                             returnVectors=True, hintLanguage='en')
        en_text = ''
        for vec in vectors:
            if vec[3] == 'en':
                en_text += text[vec[0] : vec[0]+vec[1]]
        return ('en-extract', lang_prob, en_text)
    else:
        return ('', 0, '')


# In[25]:


predicted_lang = pd.DataFrame.from_records(documents.apply(predict_language), columns=('lang', 'lang_prob', 'text'), index=documents.index)


# In[26]:


predicted_lang_en_mask = predicted_lang['lang'].isin(['en', 'en-extract'])
(~ predicted_lang_en_mask).sum()


# In[27]:


texts_en = texts.where(predicted_lang_en_mask, None)
texts_en = texts.apply(lambda x: x if x is not None else [])


# ##### Filter scispacy ents

# In[28]:


from collections import Counter
if ents is not None:
    ents_counter = Counter()
    for x in ents.iteritems():
        for w in x[1]:
            ents_counter[w] += 1
    ents_common = [k for k, c in ents_counter.items() if c >= 5]
    len(ents_common)


# ##### Extended token sets 

# In[29]:


dictionary = gensim.corpora.Dictionary(texts)
if ents is not None:
    dictionary.add_documents([ents_common])


# In[30]:


# Several combinations attempted, but 'text-ents' was most useful
if ents is not None:
#    corpora['text-ents'] = (texts + ents).apply(dictionary.doc2bow)
    corpora['text-ents-en'] = (texts_en + ents).apply(dictionary.doc2bow)


# In[31]:


corpora.keys()


# ### HTML Templates

# In[32]:


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

# In[33]:


num_topics = [80]  # number of topics


# In[34]:


cmallet = {}
for c in corpora.keys():
    cmallet[c] = {}
    for i in num_topics:
        print('Building model for %s (%s topic)' % (c,i))
        prefix = os.path.join(model_path, c, str(i), '')
        os.makedirs(prefix, mode = out_path_mode, exist_ok = True)
        cmallet[c][i] = gensim.models.wrappers.ldamallet.LdaMallet(mallet_path, corpora[c], id2word=dictionary, optimize_interval=10,
                                                            prefix=prefix, workers=model_build_workers,
                                                                   num_topics=i, iterations=2500, random_seed=random_seed)


# #### Save cmallet

# In[35]:


for c in cmallet.keys():
    for i in cmallet[c].keys():
        cmallet[c][i].save(f'{gs_model_path_prefix}gensim-mallet-model_{c}_{i}.pkl4', 
                                    separately=[], sep_limit=134217728, pickle_protocol=4)
        print(f'{gs_model_path_prefix}gensim-mallet-model_{c}_{i}.pkl4')


# ### Plot

# In[37]:


vis_data = {}
gensim_lda_model = {}
for c in cmallet.keys():
    vis_data[c] = {}
    gensim_lda_model[c] = {}
    for i in cmallet[c].keys():
        gensim_lda_model[c][i] = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(cmallet[c][i])
        vis_data[c][i] = pyLDAvis.gensim.prepare(gensim_lda_model[c][i], corpora[c], 
                                                   dictionary=cmallet[c][i].id2word, mds='tsne')
        pyLDAvis.save_json(vis_data[c][i], outdir + f'pyldavis_{c}_{i}.json')
        print(outdir + f'pyldavis_{c}_{i}.json')
        ofdir = web_out_dir + f'{c}-{i}/'
        os.makedirs(ofdir, mode = out_path_mode, exist_ok = True)   
        pyLDAvis.save_html(vis_data[c][i], ofdir + f'pyldavis_{c}_{i}.html',
                          ldavis_url=MODIFIED_LDAVIS_URL)
        print(web_out_dir + f'{c}-{i}/pyldavis_{c}_{i}.html')


# #### Save Gensim Mallet Models

# In[38]:


for c in gensim_lda_model.keys():
    for i in gensim_lda_model[c].keys():
        gensim_lda_model[c][i].save(f'{gs_model_path_prefix}gensim-lda-model_{c}_{i}.pkl4', 
                                    separately=[], sep_limit=134217728, pickle_protocol=4)
        print(f'{gs_model_path_prefix}gensim-lda-model_{c}_{i}.pkl4')


# #### Save _Relevant_ terms for topics (from pyLDAviz)

# In[39]:


num_terms = 50


# In[40]:


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


# In[41]:


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


# In[42]:


topic_lists.keys()


# In[43]:


# !pip install openpyxl


# In[44]:


# Save relevant topics - write to xlsx (one corp-numtopics per sheet)
with pd.ExcelWriter(outdir + f'topics-relevant-words-abstracts-{datafile_date}-{num_terms}terms.xlsx') as writer: 
    for sheetname, dataframe in topic_lists.items():
        dataframe.to_excel(writer, sheet_name=sheetname)
print(outdir + f'topics-relevant-words-abstracts-{datafile_date}-{num_terms}terms.xlsx')


# #### Save Relevant Topics as html

# In[45]:


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


# In[46]:


# topic_lists['text-ents-80']


# ### Create dataframes of topic model collections

# In[47]:


ctopicwords_df = {}
for c in cmallet.keys():
    ctopicwords_df[c] = {}
    for i in cmallet[c].keys():
        ctopicwords_df[c][i] = pd.read_table(cmallet[c][i].ftopickeys(), header=None, names=['id', 'weight', 'wordlist'])


# In[48]:


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


# In[49]:


# Clean words
for c in ctopicwords_df.keys():
    for i in ctopicwords_df[c].keys():
        ctopicwords_df[c][i]['wordlist'] = ctopicwords_df[c][i]['wordlist'].apply(lambda x: ' '.join(normalize_topic_words(x.split())))


# In[50]:


# set(REMOVED)


# In[51]:


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


# In[52]:


# doctopics
cdoctopics_df = {}
for c in cmallet.keys():
    cdoctopics_df[c] = {}
    for n in cmallet[c].keys():
        cdoctopics_df[c][n] = pd.read_table(cmallet[c][n].fdoctopics(), header=None, names=['id']+[i for i in range(n)])
        cdoctopics_df[c][n].drop(['id'], axis=1, inplace=True)
cdoctopics_df[c][n].head()


# In[53]:


# Reorder topics
for c in cdoctopics_df.keys():
    for n in cdoctopics_df[c].keys():
# (include top 3 topics in name)        cdoctopics_df[c][n] = cdoctopics_df[c][n].T.join(ctopicwords_df[c][n][['rank', 'label']]).set_index('label').sort_values('rank').drop(['rank'], axis=1).T
        cdoctopics_df[c][n] = cdoctopics_df[c][n].T.join(ctopicwords_df[c][n][['rank', 'topicnum']]).set_index('topicnum').sort_values('rank').drop(['rank'], axis=1).T
        cdoctopics_df[c][n].T.index.rename('topic', inplace=True)
# cdoctopics_df[c][n].head()


# ### Save documents

# In[54]:


# Save topicwords
for c in ctopicwords_df.keys():
    for i in ctopicwords_df[c].keys():
        ctopicwords_df[c][i].sort_values('rank').to_csv(outdir + 'topickeys_sorted_%s_%d.txt' % (c, i), index_label='original_order')
        print(outdir + 'topickeys_sorted_%s_%d.txt' % (c, i))
        # ctopicwords_df[c][i].sort_values('rank').to_excel('out/topickeys_sorted_%s_%d.xlsx' % (c, i), index_label='original_order')


# In[55]:


# Save doctopics
for c in cdoctopics_df.keys():
    for n in cdoctopics_df[c].keys():
        cdoctopics_df[c][n].to_csv(outdir + 'doctopic_%s_%d.csv' % (c, n), index_label='original_order')
        print(outdir + 'doctopic_%s_%d.csv' % (c, n))


# In[56]:


sims_names = ['scispacy', 'specter']
sims_columns = [f'sims_{x}_cord_uid' for x in sims_names]


# In[57]:


assert all(x in original_df.columns for x in sims_columns)


# In[58]:


assert 'cord_uid' in original_df.columns


# In[59]:


def helper_get_sims_html_ids(sim_uids, cord_uid_topic_num, cord_uid_cite_ad):
    result = []
    for uid in sim_uids:
        topic_num = cord_uid_topic_num.get(uid)
        cite_ad = cord_uid_cite_ad.get(uid)
        if cite_ad and topic_num:
            result.append(f'<a href="Topic_{topic_num}.html#{uid}">{cite_ad}</a>')
    return ', '.join(result)


# In[60]:


original_df['abstract_mentions_covid'].sum()


# In[61]:


# Prepare to save docs by topics
predominant_doc_dfd = {}
predominant_doc_df = original_df[['cite_ad', 'title', 'authors', 'publish_year', 'publish_time', 
                                  'dataset', 'abstract_mentions_covid',
                                 'pmcid', 'pubmed_id', 'doi', 'cord_uid', 'sha', 'abstract_clean']
                                 + sims_columns
                                 ].copy()
sims_mapping_cord_uid_sd = {}
predominant_doc_df['publish_time'] = predominant_doc_df['publish_time'].dt.strftime('%Y-%m-%d')
for c in cdoctopics_df.keys():
    predominant_doc_dfd[c] = {}
    sims_mapping_cord_uid_sd[c] = {}
    for n in cdoctopics_df[c].keys():
        predominant_doc_dfd[c][n] = {}
        sims_mapping_cord_uid_sd[c][n] = {}
        predominant_doc_df['predominant_topic'] = cdoctopics_df[c][n].idxmax(axis=1)
        predominant_doc_df['predominant_topic_num'] = predominant_doc_df['predominant_topic'].str.split().apply(lambda x: x[0][1:])
        predominant_doc_df['major_topics'] = cdoctopics_df[c][n].apply(lambda r: {f't{i + 1:02d}': val for i, val in enumerate(r) if val >= 0.3}, axis=1)
        for sim_col in sims_columns:
            sims_mapping_cord_uid_sd[c][n][sim_col] = {}
            sims_mapping_cord_uid_sd[c][n][sim_col]['topic_num'] = predominant_doc_df[['cord_uid', 'predominant_topic_num']].set_index('cord_uid')['predominant_topic_num']
            sims_mapping_cord_uid_sd[c][n][sim_col]['cite_ad'] = predominant_doc_df[['cord_uid', 'cite_ad']].set_index('cord_uid')['cite_ad']
        for i, topic_name in enumerate(cdoctopics_df[c][n].columns):        
            temp_df = predominant_doc_df[(predominant_doc_df['major_topics'].apply(lambda x: topic_name in x))].copy()
            temp_df['topic_weight'] = temp_df.major_topics.apply(lambda x: x.get(topic_name))
            temp_df = temp_df.sort_values(['topic_weight'], axis=0, ascending=False)
            predominant_doc_dfd[c][n][i] = temp_df


# In[62]:


# Save docs by topics - write to json and tsv
for c in predominant_doc_dfd.keys():
    for n in predominant_doc_dfd[c].keys():
        outfile_central_docs_base = outdir + f'topics-central-docs-abstracts-{datafile_date}-{c}-{n}'
        temp_dfs = []
        for i, dataframe in predominant_doc_dfd[c][n].items():
            temp_df = dataframe[['title', 'authors', 'publish_year', 'publish_time', 'cord_uid', 'dataset', 'sha', 'abstract_clean']].reset_index()
            temp_df['Topic'] = i + 1
            temp_dfs.append(temp_df)
        result_df = pd.concat(temp_dfs)
        print(outfile_central_docs_base + '.{jsonl, txt}')
        result_df.to_json(outfile_central_docs_base + '.jsonl', **out_json_args)
        result_df.to_csv(outfile_central_docs_base + '.txt', sep='\t')


# In[63]:


# Save docs by topics - write to excel
for c in predominant_doc_dfd.keys():
    for n in predominant_doc_dfd[c].keys():
        print(outdir + f'topics-central-docs-abstracts-{datafile_date}-{c}-{n}.xlsx')
        with pd.ExcelWriter(outdir + f'topics-central-docs-abstracts-{datafile_date}-{c}-{n}.xlsx') as writer: 
            for i in predominant_doc_dfd[c][n].keys():
                sheetname = f'Topic {i+1}'
                predominant_doc_dfd[c][n][i].drop(columns=['abstract_clean', 'cite_ad', 'major_topics',
                                                          'predominant_topic', 'predominant_topic_num']
                                                 ).to_excel(writer, sheet_name=sheetname)


# In[64]:


# prep similarity columns for html
for c in predominant_doc_dfd.keys():
    for n in predominant_doc_dfd[c].keys():
        for sim_name, sims_col in zip(sims_names, sims_columns):
            cord_uid_topic_num = sims_mapping_cord_uid_sd[c][n][sim_col]['topic_num'].to_dict()
            cord_uid_cite_ad = sims_mapping_cord_uid_sd[c][n][sim_col]['cite_ad'].to_dict()
            for i in predominant_doc_dfd[c][n].keys():
                predominant_doc_dfd[c][n][i][f'Similarity {sim_name}'] = (predominant_doc_dfd[c][n][i][sims_col]
                 .apply(lambda x: helper_get_sims_html_ids(x, cord_uid_topic_num, cord_uid_cite_ad)))


# In[65]:


# Modify dataframe for html
for c in predominant_doc_dfd.keys():
    for n in predominant_doc_dfd[c].keys():
        for i in predominant_doc_dfd[c][n].keys():
            predominant_doc_dfd[c][n][i]['pmcid'] = predominant_doc_dfd[c][n][i]['pmcid'].apply(lambda xid: f'<a href="https://www.ncbi.nlm.nih.gov/pmc/articles/{xid}" target="_blank">{xid}</a>' if not pd.isnull(xid) else '')
            predominant_doc_dfd[c][n][i]['pubmed_id'] = predominant_doc_dfd[c][n][i]['pubmed_id'].apply(lambda xid: f'<a href="https://www.ncbi.nlm.nih.gov/pubmed/{xid}" target="_blank">{xid}</a>' if not pd.isnull(xid) else '')
            predominant_doc_dfd[c][n][i]['doi'] = predominant_doc_dfd[c][n][i]['doi'].apply(lambda xid: f'<a href="https://doi.org/{xid}" target="_blank">{xid}</a>' if not pd.isnull(xid) else '')
            predominant_doc_dfd[c][n][i]['abstract_mentions_covid'] = predominant_doc_dfd[c][n][i]['abstract_mentions_covid'].apply(lambda x: 'Y' if x else 'N')
            predominant_doc_dfd[c][n][i].columns = [' '.join(c.split('_')) for c in predominant_doc_dfd[c][n][i].columns]


# In[66]:


from pandas.io.formats import format as fmt
from pandas.io.formats.html import HTMLFormatter
from typing import Any, Optional


# In[67]:


class MyHTMLFormatter(HTMLFormatter):
    "Add html id to th for rows"
    def __init__(self, html_id_col_name, *args, **kwargs):
        super(MyHTMLFormatter, self).__init__(*args, **kwargs)
        self.html_id_col_name = html_id_col_name

    def write_th(
        self, s: Any, header: bool = False, indent: int = 0, tags: Optional[str] = None
    ) -> None:
        if not header and self.html_id_col_name and self.html_id_col_name in self.frame.columns:
            try:
                key = int(s.strip())
            except ValueError:
                key = None
            if key and key in self.frame.index:
                html_id = self.frame.loc[key, self.html_id_col_name]
                if html_id:
                    if tags:
                        tags += f'id="{html_id}";'
                    else:
                        tags = f'id="{html_id}";'
        super(MyHTMLFormatter, self).write_th(s, header, indent, tags)


# In[68]:


# Save doc by topics - write to html
# out_topics_html_dir = outdir + f'topics-central-docs-abstracts-{datafile_date}-html/'
out_topics_html_dir = web_out_dir
os.makedirs(out_topics_html_dir, mode = out_path_mode, exist_ok = True)
for c in predominant_doc_dfd.keys():
    for n in predominant_doc_dfd[c].keys():
        ofdir = out_topics_html_dir + f'{c}-{n}/'
        os.makedirs(ofdir, mode = out_path_mode, exist_ok = True)   
        print(ofdir)
        for i in predominant_doc_dfd[c][n].keys():
            ofname = ofdir + f'Topic_{i+1:02d}.html'
            with open(ofname, 'w') as ofp:
                html_df = (predominant_doc_dfd[c][n][i]
                                .drop(columns=['sha', 'major topics', 'abstract clean',
                                              'predominant topic', 'predominant topic num'] 
                                      + [' '.join(c.split('_')) for c in sims_columns])
                                .copy()
                                .set_index(np.arange(1, len(predominant_doc_dfd[c][n][i])+1)))
                # html_table = html_df.to_html(escape=False)
                df_formatter = fmt.DataFrameFormatter(escape=False, frame=html_df, index=True, bold_rows=True)
                html_formatter = MyHTMLFormatter('cord uid', formatter=df_formatter)
                # html_formatter = HTMLFormatter(formatter=df_formatter)
                html_table = html_formatter.get_result()
                html_str = html_template.format(f'Topic {i+1:02d}', html_style, html_table)
                ofp.write(html_str)

