
# coding: utf-8

# In[ ]:

from py3k_fix import *


# In[ ]:

get_ipython().run_cell_magic('javascript', '', "IPython.keyboard_manager.command_shortcuts.add_shortcut('Ctrl-k','ipython.move-selected-cell-up')\nIPython.keyboard_manager.command_shortcuts.add_shortcut('Ctrl-j','ipython.move-selected-cell-down')\nIPython.keyboard_manager.command_shortcuts.add_shortcut('Shift-m','ipython.merge-selected-cell-with-cell-after')")


# In[ ]:

from collections import namedtuple, Counter, defaultdict, OrderedDict
from functools import wraps, partial
from glob import glob
import numpy as np
import operator as op
from operator import itemgetter as itg, attrgetter as prop, methodcaller as mc
from os.path import join
import re
from scipy import stats
import toolz.curried as z
import warnings


warnings.filterwarnings("ignore")
p = print


# pat = re.compile(r'.+? HP (\d+).+')
# for fn in glob('src/orig/*.rtf'):
#     print(fn)
#     [i] = pat.findall(fn)
#     rtfdst = join('src', 'rtf', 'hp{}.rtf'.format(i))
#     txtdst = join('src', 'txt', 'hp{}.txt'.format(i))
#     !cp "$fn" "$rtfdst"
#     !unoconv --format=txt --output=$txtdst $rtfdst

#     with open("src/txt/hp1.txt",'rb') as f:
#         txt = f.read().decode("utf-8-sig")
#     #     doc = Rtf15Reader.read(f)
#     t = txt[:60000]

# # Load and Clean Text

# In[ ]:

# with open('src/nltk_stopwords.txt', 'r') as f:
with open('src/stops.txt', 'r') as f:
    stops = set(l for l in f.read().splitlines() if l and not l.startswith('#'))


# In[ ]:

Chapter = namedtuple('Chapter', 'num title text')


class Book(object):
    def __init__(self, title, chapters: {int: Chapter}):
        self.chapters = chapters
        self.title = title
        self.txts = OrderedDict()
        for n, chap in sorted(chapters.items()):
            setattr(self, 't{}'.format(n), chap.text)
            self.txts[n] = chap.text
        txt = reduce(op.add, self.txts.values())
        self.txt = clean_text(txt)


class BookSeries(object):
    def __init__(self, n=7):
        bks = {i: parsebook(i, vb=1) for i in range(1, n + 1)}
        
        self.txts = OrderedDict()
        for n, bk in sorted(bks.items()):
            setattr(self, 'b{}'.format(n), bk.txt)
            self.txts[n] = bk.txt
        txt = reduce(op.add, self.txts.values())
        self.txt = clean_text(txt)


# In[ ]:

bookpat = re.compile(r'''\A(?P<title>.+)
(?!(.+)+?)
(?P<body>(Chapter 1)\n+(.+\n+)+)''')

bookpat = re.compile(r'''\A(?P<title>.+)
\n*
(?:(?:.+\n+)+?)
(?P<body>
    (Chapter\ 1)
    \n+
    (.+\n*)+
)''', re.VERBOSE)

chappat = re.compile(r'''(Chapter (\d+)\n+((?:.+\n+)+))+''')
chapsep = re.compile(r'Chapter (\d+)\n(.+)\n+')
# m = bookpat.match(t)
# gd = m.groupdict()
# title = gd['title']
# body = gd['body']
# gd
# m
# m.groupdict()


# book = {int(chnum): Chapter(int(chnum), title, text) for chnum, title, text in z.partition(3, chs)}
# bk = Book(book)
# book

# In[ ]:

def parsebook(fn="src/txt/hp1.txt", vb=False):
    global txt
    p = print if vb else (lambda *x, **y: None)
    if isinstance(fn, int):
        fn = "src/txt/hp{}.txt".format(fn)
    p('Reading {}'.format(fn)) 
    with open(fn,'rb') as f:
        txt = f.read().decode("utf-8-sig")
        
    gd = bookpat.search(txt).groupdict()
    
    booktitle = gd['title']
    body = gd['body']

    chs = chapsep.split(body)[1:]
    book = {int(chnum): Chapter(int(chnum), title, text) for chnum, title, text in z.partition(3, chs)}
    return Book(booktitle, book)


def clean_text(t):
    reps = {
        '’': "'",
        '‘': "'",
        '“': '"',
        '”': '"',
        '\xad': '',
        '—': '-',
       }
    
    def rep(s, frto):
        fr, to = frto
        return s.replace(fr, to)
    t = reduce(rep, reps.items(), t)
    return t

# bk = parsebook()
bks = BookSeries(5)
bksall = BookSeries(7)
# bk = bks.b1


# In[ ]:

print(bk.txt[:2000])


# In[ ]:

Counter(filter(lambda x: not re.match(r'[\dA-Za-z \."\'\-\(\);:!\?,\n]', x), bks.b1))


# In[ ]:

Complexity
- avg word length (syllables?)
- avg sentence length

- Words: 129 | Syllables: 173 | Sentences: 7 | Characters: 568 | Adverbs: 4
Characters/Word: 4.4 | Words/Sentence: 18.4


# # Text

# In[ ]:

from pandas import DataFrame, Series
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.options.display.notebook_repr_html = False
pd.options.display.width = 120
get_ipython().magic('matplotlib inline')


# In[ ]:

import spacy.parts_of_speech as ps
import spacy.en
from spacy.en import English

nlp = English()


# In[ ]:

tokens = nlp(bks.b1, tag=True, parse=True, entity=True)


# In[ ]:

bktks = {i: nlp(bktxt, tag=True, parse=True, entity=True) for i, bktxt in bks.txts.items()}
bktksall = {i: nlp(bktxt, tag=True, parse=True, entity=True) for i, bktxt in bksall.txts.items()}


# In[ ]:

def tobooks(f, bks=bktks):
    return pd.concat([f(v, i) for i, v in bks.items()])

def booker(f: 'toks -> [str]') -> '(toks, int) -> DataFrame':
    @wraps(f)
    def tobookdf(toks, bknum):
        df = DataFrame(f(toks), columns=['Val'])
        df['Book'] = bknum
        return df
    return tobookdf
    
over_books = z.comp(partial(tobooks, bks=bktksall), booker)

sent_lens = booker(lambda parsed: [spanlen(sent) for sent in parsed.sents])
wd_lens = booker(lambda parsed: [len(tok) for tok in parsed if len(tok) > 1])
spanlen = lambda span: len([wd for wd in span if len(wd) > 1])


# In[ ]:

wd_len = tobooks(wd_lens, bks=bktksall)
wd_len.groupby('Book')['Val'].agg(['mean', 'median', 'std'])


# In[ ]:

sent_len = tobooks(sent_lens, bks=bktksall)
sent_len.groupby('Book')['Val'].agg(['mean', 'median', 'std'])


# - There doesn't seem to be a discernible difference in the average word length between the books. This could be because even complex language is largely composed of shorter, commoner words, highlighted by rarer, more complex words. A way to test this could be to somehow get a measure of the frequency of rarer words

# In[ ]:

tt = bktks[1]


# In[ ]:

def reg_words(parsed):
    "Non-capitalized words > 3 chars long that aren't stopwords"
    wds = [tok.orth_ for tok in parsed]
    #wds = [tok.string.rstrip() for tok in parsed]
    return [w for w in wds if len(w) > 3 and (w.lower() not in stops) and not w[0].isupper()]

def wd_freqs(parsed):
    vcs = Series(Counter(reg_words(parsed))).sort_values(ascending=False)
    return vcs
    return vcs[vcs < 20]


# The folllowing shows the relative word frequency distribution. The first two numbers in the first column indicate that for book one, words appearing only 1 time account for 45.2% of all the word occurences, while words appearing twice account for 16.9%. If anything, it appears that the share of rare words (those appearing only once or twice) decrease with each book, rather than increase.

# In[ ]:

uncwds = over_books(wd_freqs).reset_index(drop=1)


# In[ ]:

k = 10
wdfreq = DataFrame({bknum: gdf.Val.value_counts(normalize=1)[:k]
            for bknum, gdf in uncwds.groupby(['Book'])})
wdfreq = (wdfreq * 100).round(1)
wdfreq.columns.name, wdfreq.index.name = 'Book', 'Word_freq'
wdfreq


# The cumulative share of words appearing 10 times or less also doesn't seem to indicate an increasing share of uncommon words, and if anything points to uncommon words being used more in the first three books, and deacreasing for the last four.
# 
# **artifact of fewer pages**?

# In[ ]:

wdfreq.apply(mc('cumsum')).ix[10].plot()


# This, however, only counts words that are rare within the context of this book. spaCy provides a log-probability score for each parsed word, based on its frequency in external corpora. These will be negative numbers such that a lower score indicates that a word is less common.

# In[ ]:

probs = lambda x: [tok.prob for tok in x if tok.is_lower]
# probs = listify(z.map(prop('prob')))
prob_books = over_books(probs)
# probs = z.comp(list, z.map)(itg('prob'))


# In[ ]:

def percentile(q: float) -> "[float] -> int":
    def f(s):
        return np.percentile(s, q)
    f.__name__ = 'perc_{}'.format(q)
    return f


# In[ ]:

_probstats = prob_books.groupby('Book').Val.agg(['mean', 'std', 'median', percentile(5), percentile(25), ])
_probstats


# In[ ]:

_probstats[['perc_5', 'perc_25', 'median', 'mean']].plot();


# In[ ]:

Summarizing the basic statistics of the uncommonness of the words, we see a slight dip between the first and second books


# In[ ]:

prob_books[:3]


# In[ ]:

probs(bktks[2])


# In[ ]:

v = prob_books.query('Val < -15.2 & Val > -16 & Book == 2').Val
v.value_counts(normalize=0)


# In[ ]:

plt.figure(figsize=(16, 10))
pt = sns.boxplot
pt = sns.violinplot
pp = prob_books.query('Val < -14')
pt(x='Book', y='Val', data=pp)


# In[ ]:

plt.figure(figsize=(16, 10))

for k, gdf in prob_books.groupby('Book'):
    gdf.query('Val < -8').Val.hist(bins=200, alpha=.2, normed=True)
    
    if k > 6:
        break
        
plt.legend(range(1, 6))


# In[ ]:

newix = [range(length) for bk, length in prob_books.Book.value_counts(normalize=0).sort_index().items()]
pb2 = prob_books.copy()
pb2['Ix2'] = np.concatenate(newix)


# In[ ]:

pb2.pivot(index='Ix2', columns='Book', values='Val')


# In[ ]:

prob_books[:5]


# In[ ]:

prob_books.unstack()


# In[ ]:

def plot_cumsum(s, **kw):
    cmsm = (s.sort_values(ascending=True)
            .reset_index().reset_index()
            .set_index('Val', drop=False)['level_0'].cumsum())
    cmsm /= cmsm.max()
    cmsm.plot()


# In[ ]:

gdf.Val.sort_values(ascending=True).describe()


# In[ ]:

plt.figure(figsize=(16, 10))  # 196000
g = sns.FacetGrid(prob_books[:].query('Val < -14'), row="Book", aspect=8, size=2)
g.map(plot_cumsum, 'Val'); None


# In[ ]:

plt.figure(figsize=(16, 10))  # 196000
g = sns.FacetGrid(prob_books[:].query('Val < -8'), row="Book", aspect=8, size=2)
g.map(plt.hist, 'Val', normed=True, bins=200); None


# In[ ]:




# In[ ]:

prob_books.Val.kurtosis()


# In[ ]:


for k, gdf in prob_books.groupby('Book'):
    gdf.query('Val < -8').Val.hist(bins=200, alpha=.2, normed=True)
    
    if k > 6:
        break
        
plt.legend(range(1, 6))


# In[ ]:




# In[ ]:

# pbs = probs(tokens)
pbs = prob_books(bktks)


# In[ ]:

map(z.comp(str.strip, str), bktks[3])


# In[ ]:

get_words = listify(z.map(z.comp(str.strip, str)))


# In[ ]:

_tks = bktks[2]
pbdf = (DataFrame(zip(probs(_tks), get_words(_tks)))
        .sort_values(0, ascending=False)
        .rename(columns={0: 'Prob', 1: 'Word'})
       .assign(Prob=lambda x: x.Prob.round(3)))


# In[ ]:

# pbdf.query('Prob == -15.516')  # .Word.value_counts(normalize=0)
pbdf.query('Prob == -10.963').Word.value_counts(normalize=0)


# In[ ]:

pbdf.query('Prob < -8 & Prob > -16').Prob.value_counts(normalize=0)


# In[ ]:

pbdf.query('Prob < -15.2 & Prob > -16').Prob.value_counts(normalize=0)


# In[ ]:

get_ipython().system('open /tmp/x.csv')


# In[ ]:

import numpy.random as nr
import pandas as pd
nr.seed(10)
a = nr.randint(0, 20, (2, 3))
df = pd.DataFrame(a)
df


# In[ ]:

df.index.name = 'A0'
df.columns.name = 'A1'


# In[ ]:

get_ipython().magic('pinfo df.rename')


# In[ ]:

This count is only a 


# In[ ]:

for tok in tokens:
    break


# In[ ]:

tok.prob


# In[ ]:


gdf.Val.value_counts(normalize=1)[:k]


# In[ ]:

uncwds.groupby(['Book', 'Val']).size()


# In[ ]:

uncwds.Book.value_counts(normalize=0)


# In[ ]:

uncwds[:5]


# In[ ]:

1


# In[ ]:

reg_words


# In[ ]:

wd_freqs(tt)


# In[ ]:

vcs = Series(Counter(reg_words(tt))).sort_values(ascending=False)
bm = vcs.index.map(lambda x: len(x) > 3 and (x.lower() not in stops) and not x[0].isupper())
vcs = vcs[bm]


# In[ ]:

R = 19
DataFrame(list(z.partition(R, vcs.index[:R*20])))


# In[ ]:

# Longest words
Series([tok.string.strip() for bk in bktks.values() for tok in bk if len(tok) > 20]).value_counts(normalize=0)


# In[ ]:

sent_lens = pd.concat([sent_lens(v, i) for i, v in bktks.items()])


# In[ ]:

# Longest sentences
Series().value_counts(normalize=0)


# In[ ]:

for s in [sent for bk in bktks.values() for sent in bk.sents if spanlen(sent) > 200]:
    print(s, end='\n=======\n')


# In[ ]:

[tok for tok in bktks[5] if len(tok) > 40]


# In[ ]:

plt.figure(figsize=(16, 10))
sns.boxplot(x='Book', y=0, data=wd_lens)


# In[ ]:

plt.figure(figsize=(16, 10))
sns.boxplot(x='Book', y=0, data=sent_lens)


# In[ ]:

plt.figure(figsize=(16, 10))
pt = sns.boxplot
pt = sns.violinplot
pt(x='Book', y=0, data=uncwds)
# plt.ylim(0, 20)


# In[ ]:

sent_lens.groupby('Book')[0].median()


# In[ ]:




# In[ ]:

sent_lens


# In[ ]:

z.valmap()


# In[ ]:

def 


# In[ ]:

DataFrame(z.valmap(sent_lens, bktks))


# In[ ]:

bktks[1]


# In[ ]:

sent_lens(tks)


# In[ ]:

span.end - span.start


# In[ ]:




# In[ ]:

len(wd)


# In[ ]:

for wd in list(span):
    wd


# In[ ]:

for span in tokens.sents:
    break


# In[ ]:

span.text_with_ws


# In[ ]:

tokens.


# In[ ]:

len(list(tokens.sents))


# In[ ]:

span.end


# In[ ]:

[tokens[i] for i in range(span.start, span.end)]


# In[ ]:

span.start


# In[ ]:

tok


# In[ ]:

tok.ent_iob


# In[ ]:

people = Series(ent.string.rstrip() for ent in tokens.ents if ent.label_ == 'PERSON')


# In[ ]:

pn = people.value_counts(normalize=0)
pn


# In[ ]:

pn[40:-40]


# In[ ]:

for ent in tokens.ents:
    print(ent, ent.label_)


# In[ ]:

for tok in tokens[:40]:
    print(tok, ps.NAMES[tok.pos])
    1


# #Find Characters

# In[ ]:

wds = tokens


# In[ ]:

caps = Series([tok.string.rstrip() for tok in tokens if tok.is_title and tok.pos == ps.NOUN])


# In[ ]:

import pandas as pd
from pandas import Series, DataFrame


# In[ ]:

caps.


# In[ ]:

caps.value_counts(normalize=0)


# In[ ]:

tok.is_title


# In[ ]:

ps.PROPN


# In[ ]:

for pos in dir(ps)[8:]:
    print(pos)
    if pos.isupper():
        int(getattr(ps, pos))


# In[ ]:




# In[ ]:

[getattr(ps, pos) for pos in dir(ps) if pos.isupper()]


# In[ ]:

{getattr(ps, pos): pos for pos in dir(ps) if pos.isupper()}


# In[ ]:

tok


# In[ ]:

tok.pos


# In[ ]:

tokens


# In[ ]:

get_ipython().system('open /Users/williambeard/miniconda3/envs/hp/lib/python3.5/site-packages/spacy/en/')

