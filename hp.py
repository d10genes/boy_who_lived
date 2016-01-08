
# coding: utf-8

# In[ ]:

from py3k_fix import *


# In[ ]:

get_ipython().run_cell_magic('javascript', '', "IPython.keyboard_manager.command_shortcuts.add_shortcut('Ctrl-k','ipython.move-selected-cell-up')\nIPython.keyboard_manager.command_shortcuts.add_shortcut('Ctrl-j','ipython.move-selected-cell-down')\nIPython.keyboard_manager.command_shortcuts.add_shortcut('Shift-m','ipython.merge-selected-cell-with-cell-after')")


# In[ ]:

from collections import namedtuple, Counter, defaultdict, OrderedDict
from functools import wraps, partial
from glob import glob
from itertools import count
import operator as op
from operator import itemgetter as itg, attrgetter as prop, methodcaller as mc
from os.path import join
import re
import sys
import warnings

from joblib import Parallel, delayed
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame, Series
import pandas as pd
from scipy import stats
import seaborn as sns
import toolz.curried as z

from IPython.display import Image

warnings.filterwarnings("ignore")
p = print
pd.options.display.notebook_repr_html = False
pd.options.display.width = 120
get_ipython().magic('matplotlib inline')


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
        bks = {i: parsebook(i, vb=False) for i in range(1, n + 1)}
        
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

Counter(filter(lambda x: not re.match(r'[\dA-Za-z \."\'\-\(\);:!\?,\n]', x), bks.b1))


#     Complexity
#     - avg word length (syllables?)
#     - avg sentence length
# 
#     - Words: 129 | Syllables: 173 | Sentences: 7 | Characters: 568 | Adverbs: 4
#     Characters/Word: 4.4 | Words/Sentence: 18.4
#     - sentence structure

# # Parse Text
# I have recently come across the [spaCy](https://spacy.io) library, which bills itself as a "library for industrial-strength natural language processing in Python and Cython," and this seemd like a good opportunity to explore its capabilities. The starting point is a parsing function that parses, tags and detects entities all in one go.

# In[ ]:

# import spacy.parts_of_speech as ps
# import spacy.en
from spacy.en import English

nlp = English()


# In[ ]:

bktks = {i: nlp(bktxt, tag=True, parse=True, entity=True) for i, bktxt in bks.txts.items()}
bktksall = {i: nlp(bktxt, tag=True, parse=True, entity=True) for i, bktxt in bksall.txts.items()}


# In[ ]:

def tobooks(f: '(toks, int) -> DataFrame', bks=bktks) -> DataFrame:
    """Apply a function `f` to all the tokens in each book,
    putting the results into a DataFrame column, and adding
    a column to indicate each book.
    """
    return pd.concat([f(v, i) for i, v in bks.items()])

def booker(f: 'toks -> [str]') -> '(toks, int) -> DataFrame':
    @wraps(f)
    def tobookdf(toks, bknum):
        df = DataFrame(f(toks), columns=['Val'])
        df['Book'] = bknum
        return df
    return tobookdf
    
over_books = z.comp(partial(tobooks, bks=bktksall), booker)


# # Search for increasing complexity
# ## Average word and sentence length
# 
# A first simple search would be to see if the average length of the words or sentences increases throughout the series.

# In[ ]:

sent_lens = booker(lambda parsed: [spanlen(sent) for sent in parsed.sents])
wd_lens = booker(lambda parsed: [len(tok) for tok in parsed if len(tok) > 1])
spanlen = lambda span: len([wd for wd in span if len(wd) > 1])


# In[ ]:

def wd_sent_lens():
    def agg_lens(lns):
        return (tobooks(lns, bks=bktksall)
              .groupby('Book')['Val'].agg(['mean', 'median', 'std'])
              .rename(columns=str.capitalize))

    wd_len = agg_lens(wd_lens)
    sent_len = agg_lens(sent_lens)
    
    lens = {'Sentence_length': sent_len, 'Word_length': wd_len}
    return pd.concat(lens.values(), axis=1, keys=lens.keys())
 
wsls = wd_sent_lens()
wsls


# In[ ]:

plt.figure(figsize=(16, 5))
plt.subplot(1, 2, 1)
wsls.Word_length['Mean'].plot(title='Average word length')
plt.subplot(1, 2, 2)
wsls.Sentence_length['Mean'].plot(title='Average sentence length');


# There does appear to be an increasing trend for both average word and sentence length difference between the books, though the scale of the difference is miniscule in light of the standard deviations.

# In[ ]:

# tt = bktks[1]


# ## Word complexity by frequency
# 
# The lack of discernible difference in word/sentence length could be because even complex language is still largely composed of shorter, commoner words, highlighted by rarer, more complex words. A way to test this could be to somehow get a measure of the frequency of just the rarer words by counting, for example,  what percentage of the words only appear once.

# In[ ]:

def reg_words(parsed):
    "Non-capitalized words > 3 chars long that aren't stopwords"
    wds = [tok.orth_ for tok in parsed]
    #wds = [tok.string.rstrip() for tok in parsed]
    return [w for w in wds if len(w) > 3 and (w.lower() not in stops) and not w[0].isupper()]

def wd_freqs(parsed):
    vcs = Series(Counter(reg_words(parsed))).sort_values(ascending=False)
    return vcs


# In[ ]:

uncwds = over_books(wd_freqs).reset_index(drop=1)


# The folllowing shows the relative word frequency distribution. The first two numbers in the first column indicate that for book one, words appearing only 1 time account for 45.2% of all the word occurences, while words appearing twice account for 16.9%. If anything, it appears that the share of rare words (those appearing only once or twice) *decreases* with each book, rather than increases.

# In[ ]:

k = 10
wdfreq = DataFrame({bknum: gdf.Val.value_counts(normalize=1)[:k]
            for bknum, gdf in uncwds.groupby(['Book'])})
wdfreq = (wdfreq * 100).round(1)
wdfreq.columns.name, wdfreq.index.name = 'Book', 'Word_freq'
wdfreq


# The cumulative share of words appearing 10 times or less also doesn't seem to indicate an increasing share of uncommon words, and if anything points to uncommon words being used more in the first three books, and deacreasing for the last four. (The following graph should be interpreted to say that, for example, 90% of the words in the first book are those that appear fewer than 11 times, while 86% of the words in book 5 occur fewer than 11 times).

# In[ ]:

wdfreq.apply(mc('cumsum')).ix[10].plot()
plt.ylabel('% of words in each book\n that appear 10 times or less');


# ## Word complexity by frequency in English language usage

# The frequency counting above, however, only counts words that are rare within the context of this series. Fortunateky, spaCy provides a log-probability score for each parsed word, based on its frequency in external corpora. These will be negative numbers such that a lower score indicates that a word is less common in English usage outside of Harry Potter. "Low probability," "low likelihood" and "less common" are terms I'll use to describe words with low log-probability scores.

# In[ ]:

probs = lambda x: [tok.prob for tok in x if tok.is_lower]
prob_books = over_books(probs)


# In[ ]:

def percentile(q: float) -> "[float] -> int":
    def f(s):
        return np.percentile(s, q)
    f.__name__ = 'perc_{}'.format(q)
    return f

def show_freq(bookstats):
    probstats = (bookstats.groupby('Book').Val
                 .agg(['mean', 'std', 'median',
                       percentile(5), percentile(25)])
                .rename(columns=str.capitalize))
    probstats[['Perc_5', 'Perc_25', 'Median', 'Mean']].plot(title='Word Frequency')
    plt.xticks(range(1, 8));
    return probstats


# In[ ]:

show_freq(prob_books)


# The most drastic difference is in the frequency of the 95th percentile between first and second books. The graph shows that a typical word in the 95th percentile has a log probability of -13.3 in the first book and -13.8 in the second. The drop doesn't look that drastic, and there doesn't seem to be a discernable overall trend, either.
# 
# Out of curiosity, it could be helpful to dig into what the probabilities look like for the first couple hundred least likely words.

# In[ ]:

probs1 = probs(bktks[1])
probs2 = probs(bktks[2])
# probs12 = probs1 + probs2


# In[ ]:

def show_unc_word_trend():
    n = 200
    s1 = Series(probs1).sort_values(ascending=True).reset_index(drop=1)
    s1[s1 < -12][:n].plot()
    s2 = Series(probs2).sort_values(ascending=True).reset_index(drop=1)
    s2[s2 < -12][:n].plot(title='Log probability for $n$ rarest words')
    plt.legend(['Book 1', 'Book 2'])
    
plt.figure(figsize=(8, 5))
show_unc_word_trend()
xo, xi = plt.xlim()
plt.hlines([-18.25], xo , xi, linestyles='dashdot')
plt.hlines([-18.32], xo , xi, linestyles='dashdot');


# Starting from the least common words, it looks like the part of the reason Book 2's words are less frequent is due to a few streaks of words that have log probabilities indicated by the dashed lines. The repetition of certain uncommon words in the story line could lead us to classify some text as more complex than we should. A solution would be to run the same plots on the probabilities of *unique* words in the texts.

# In[ ]:

def get_prob_id(toks) -> 'DataFrame[Prob, Id]':
    return DataFrame([(tok.prob, tok.orth) for tok in toks if tok.is_lower], columns=['Prob', 'Id'])
 
def unique_probs(toks):
    "Like `probs`, but drop duplicate words"
    df = get_prob_id(toks)
    return df.drop_duplicates('Id').Prob.tolist()

uprob_books = over_books(unique_probs)


# In[ ]:

ufreq = show_freq(uprob_books)
ufreq


# Here, the trend towards more complex words is much more pronounced, and looks as if it continues throughout the whole series, with book 5 having disproportionately many more complex words. As anyone who's read the series can tell, Book 5 (*Order of the Phoenix*) also stands out as being disproportionately longer in page numbers, as confirmed by the wordcount:

# In[ ]:

plt.figure(figsize=(16, 5))
wc = Series(z.valmap(len, bktksall))
plt.subplot(1, 2, 1)
wc.plot(title='Word count'); plt.ylim(0, None);
plt.subplot(1, 2, 2)
plt.scatter(wc, ufreq.Mean);
plt.title('Word likelihood vs word count')
plt.ylabel('Mean log p'); plt.xlabel('Total word count');


# ...which could lead us to wonder whether the increasing complexity in word choice is simply an artifact of the length of the books (if the text were generated randomly from the same distribution, we would expect longer texts to include a greater number of unique and rarer words).

# In[ ]:

def plot_corrcoef(x=None, y=None, data=None):
    sns.regplot(x=x, y=y, data=data, order=1)
    plt.title('Corr. Coef.: {:.3f}'.format(stats.pearsonr(data[x], data[y])[0]))
    plt.ylabel('Mean log p')
    plt.xlabel('Total word count');
    
plot_corrcoef(x='Word_count', y='Mean', data=ufreq.assign(Word_count=wc))
# sns.regplot(x='Word_count', y='Mean', data=ufreq.assign(Word_count=wc))
# plt.title('Corr. Coef.: {:.3f}'.format(stats.pearsonr(ufreq.Mean, wc)[0]));


# Indeed, the relationship between typical word appears to have a quite [log] linear relationship with word count. I'm not sure what relationship is to be expected, but it looks like it would be worthwhile to try and correct for document length in determining word complexity. 

# In[ ]:

import numpy.random as nr


# In[ ]:

def simgrowth(toks, nsims=20):
    def simgrowth_():
        s = set()
        l = []
        tks = map(prop('orth'), toks)
        nr.shuffle(tks)
        for w in tks:
            s.add(w)
            l.append(len(s))
        return l
    return [simgrowth_() for _ in range(nsims)]

ls = simgrowth(bktks[1])


# In[ ]:

for l in ls:
    plt.plot(l, alpha=.05)


# ls5 = simgrowth(bktks[5])
# plt.figure(figsize=(16, 10))
# for l in ls5:
#     plt.plot(l, alpha=.05)
#     
# for l in ls:
#     plt.plot(l, alpha=.05)

# ### Simulate word distributions
# For each booklength $L$, I'll be repeatedly sampling $L$ words with replacement from the book with the largest word count, book 5, and then finding the average word probability of each sample. This should give an estimate of what the average word count should be for each book, they were all drawing from the same source, given the length of each book. 

# In[ ]:

def sim(df, seed=None, aggfunc=None, size=None, rep=False):
    dd = (df.sample(n=size, replace=rep, random_state=seed
                   ).drop_duplicates('Id').Prob)
    # with replacement, the distribution gets biased
    # towards more low-probability words
    return aggfunc(dd)


# In[ ]:

def sim_gen_text(worddist=5, sizebook=1, nsims=10000,
                 aggfunc=np.median, n_jobs=0, vb=False, rep=False):
    pt = print if vb else (lambda *x, **_: None)
    sizedf = get_prob_id(bktksall[sizebook])
    size = len(sizedf)
    if worddist == 8:
        df = pd.concat([get_prob_id(bktksall[i]) for i in range(1, 8)])
    else:
        df = get_prob_id(bktksall[worddist])
    
    mu = aggfunc(df.drop_duplicates('Id').Prob)
    pt(mu)
    if (len(df) == size) and not rep:
        return [mu for _ in range(nsims)]
        
    if len(df) < size:
        raise ValueError("Can't sample with replacement"
                         " from smaller distribution")
    f = delayed(sim) if n_jobs else sim
    gen = (f(df, seed=seed, aggfunc=aggfunc, size=size, rep=rep) for seed in range(nsims))
    if n_jobs:
        pt('Running {} jobs...'.format(n_jobs), end=' ')
        ret = Parallel(n_jobs=n_jobs)(gen)
    else:
        ret = list(gen)
    pt('Done.')
    sys.stdout.flush()
    return ret 


# %time x = sim_gen_text(worddist=5, sizebook=5, nsims=100, aggfunc=np.mean, rep=True)

# In[ ]:

def get_gen_prob_text(nsims=10000, n_jobs=-1, worddist=5, usecache=True,
                      rep=False):
    """Run simulations and cache them (running ~10k sims
    for each book) takes ~10 min."""
    fn = 'cache/gen_prob_text_{}_{}_{}.csv'.format(worddist, rep, nsims)
    
    def gen_prob_text_read():
        return pd.read_csv(fn).rename(columns=int)

    if usecache:
        try:
            return gen_prob_text_read()
        except IOError:
            print('{} not found, running simulations...'.format(fn))
            sys.stdout.flush()
    
    gens_mus = {
        booknum: sim_gen_text(worddist=worddist, sizebook=booknum,
                              nsims=nsims, aggfunc=np.mean, n_jobs=n_jobs, rep=rep)
        for booknum in range(1, 8)
    }
    d = DataFrame(gens_mus)
    d.to_csv(fn, index=None)
    return gen_prob_text_read()
    


# In[ ]:

get_ipython().magic('time d1 = get_gen_prob_text(nsims=10000, worddist=8, n_jobs=-1, rep=False)')


# %time d1 = get_gen_prob_text(nsims=20000, n_jobs=-1)

# In[ ]:

cols = ['Val', 'Book', 'Source']
d = d1.copy()
d.columns.name = 'Book'
d = d.stack().sort_index(level='Book').reset_index(drop=0).rename(columns={0: 'Val'}).drop('level_0', axis=1)
d['Source'] = 'Simulation'
dboth = d[cols].append(uprob_books.assign(Source='Actual')[cols]).sort_values(['Book', 'Source'], ascending=True)


# In[ ]:

bothagg = dboth.groupby(['Source', 'Book',]).mean()
bothagg.unstack('Source')


# In[ ]:

plt.figure(figsize=(16, 10))
pbothagg = bothagg.ix['Actual'].copy().rename(columns={'Val': 'Actual'})
pbothagg.index -= 1
plt.scatter([], [], s=80, c='k', marker='x', linewidth=2)
plt.legend(['Actual']);
sns.violinplot('Book', 'Val', data=d)
plt.scatter(pbothagg.index, pbothagg.Actual, s=80, c='k', marker='x', linewidth=2);


# Barring some subtle errors in my simulation code (which would not surprise me at all), the violin plot above says that the actual average word probability for books 2, 5, 6 and 7 are roughly what one would expect if words were drawn at random from the whole series, based solely on the length of the book. Measuring word complexity as having a low probability, this could lead one to say that the word complexity of the first book is way below average, and the word complexity if the 3rd and 4th books are somewhat below average, with 5, 6 and 7 increasingly approaching the average. This seems to be the best evidence so far of the writing complexity increasing as Harry Potter's education progresses.
# 
# The trend in increasing complexity may be clearer by plotting this difference in simulated and actual average probability:

# In[ ]:

plt.figure(figsize=(16, 5))
plt.subplot(1, 2, 1)
bothagg.unstack('Source')['Val'].eval('Simulation - Actual').plot(title='Average - actual word complexity');
plt.subplot(1, 2, 2)
plot_corrcoef(x='Word_count', y='Mean', data=d.groupby(['Book']).mean().rename(columns={'Val': 'Mean'}).assign(Word_count=wc))


# To the right, we also see that at least the simulated values are much better estimated by a linear word count predictor (negative correlation coefficient of .985 for the simulated vs .935 for the actual averages). 

# In[ ]:

DataFrame(Series({k: ((v.Val < ufreq.Mean[k]).mean() * 100).round(1) for k, v in d.groupby('Book')})).T


# ## Sentence structure complexity

# In[ ]:

import networkx as nx
import pygraphviz as pgv


# In[ ]:

s = next(bktksall[1].sents)


# G = nx.DiGraph()
# G.add_edge
# ts = bktksall[1]
# for s in ts.sents:
#     break
# s.root.head is s.root
# list(s.root.children)
# s
# s.root.
# child.n_lefts

# In[ ]:

There are different ways to measure the complexity of a sentence based on the syntactical structure, 


# In[ ]:

for a in [a for a in dir(w1) if not a.startswith('__')]:
    f1, f2 = getattr(w1, a), getattr(w2, a)
    if callable(f1):
        try:
            eq = f1() == f2()
            print(a, eq)
        except TypeError:
            pass
    else:
        print(a, f1 == f2)
    


# In[ ]:

def dedupe_wrd_repr(s):
    d = {}
    dfd = defaultdict(int)
    for tok in s:
        dfd[tok.orth_] += 1
        n = dfd[tok.orth_]
        print(tok.i, tok, n)
        if n > 1:
            d['{}[{}]'.format(tok.orth_, n)] = tok.i
        else:
            d[tok.orth_] = tok.i
    return {v: k for k, v in d.items()}

dd = dedupe_wrd_repr(s)


# In[ ]:

def add_edge(src, dst, G, reprdct=None, ):  # seen=set()
    """Since this is a tree, append an underscore for duplicate
    destination nodes"""
    s1, s2 = src.orth_, dst.orth_
    s1, s2 = src.i, dst.i
#     if s2 in seen:
#         s2 = '{}_'.format(s2)
#         seen.add(s2)
    return G.add_edge(reprdct[s1], reprdct[s2])


# In[ ]:

def add_int_edge(tok, c, G, **_):
    G.add_edge(tok.i, c.i)


# In[ ]:

def build_graph(tok, G, i=0, vb=False, reprdct=None, add_edge=add_edge):
    pp = print if vb else (lambda *x, **k: None)
#     if seen is None:
#         seen = {tok.orth_}
    pp(' ' * i, tok)
        
    for c in tok.children:
        add_edge(tok, c, G, reprdct=reprdct)
        build_graph(c, G, i=i + 2, reprdct=reprdct, add_edge=add_edge)
    return G

G = build_graph(s.root, pgv.AGraph(directed=True), vb=0, reprdct=dd)
Gi = build_graph(s.root, pgv.AGraph(directed=True), vb=0, reprdct=dd, add_edge=add_int_edge)


# In[ ]:

G.draw("file.png", prog='dot')
print(s)
Image(filename="file.png") 


# In[ ]:

def tree_depths(s):
    def tree_depths_(tok, i=1, vb=False,):
        return [i] + [t for c in tok.children
                      for t in tree_depths_(c, i=i + 1, vb=vb)]
        #return [(i, tok)]
    return tree_depths_(s.root, i=1)

# tree_depths(s)


# In[ ]:




# #Benchmark Joblib

# In[ ]:

import time

def timer(f):
    @wraps(f)
    def f2(*a, **kw):
        st = time.time()
        ret = f(*a, **kw)
        return time.time() - st, ret
    return f2

def run_timed(n_jobs=None, sims=[]):
    return [timer(get_gen_prob_text)(nsims=nsims, n_jobs=n_jobs, usecache=False)[0] for nsims in sims]

simtrials = [10, 20, 100, 300, 1000, 2000, 5000, 8000, 10000, 15000, 20000]


# In[ ]:

def benchmark(fn='cache/benchmarks.csv'):
    try:
        return pd.read_csv(fn, index_col =0).rename(columns=int)
    except IOError:
        print('No file {}, running sims'.format(fn))

    nj0 = run_timed(n_jobs=0, sims=simtrials)
    nj2 = run_timed(n_jobs=2, sims=simtrials)
    njn1 = run_timed(n_jobs=-1, sims=simtrials)
    return DataFrame(OrderedDict([(0, nj0), (2, nj2), (-1, njn1)]), index=simtrials)


# In[ ]:

benchmarks.to_csv('cache/benchmarks.csv')


# In[ ]:

benchmarks2 = benchmark()
benchmarks2


# In[ ]:

pd.util.testing.assert_frame_equal(benchmarks, benchmarks2)


# In[ ]:

benchmarks2.plot(style='-o')


# In[ ]:

mins = (benchmarks2 // 60 + (benchmarks2 % 60) * .01).round(2)
mins


# In[ ]:

get_ipython().system('say done')


# In[ ]:

DataFrame(ufreq.Median).assign(Word_count=lambda x: wc)


# In[ ]:

for k, gdf in uprob_books.groupby('Book'):
    break


# In[ ]:

nsamp = 1000
nr.seed(nsamp)
pd.np.random.seed(nsamp)
gdf.Val.sample(nsamp, replace=nsamp, random_state=1).median()


# In[ ]:

def bootstrap_wd_prob(probs: Series, nsims=100, nsamp=1000):
    meds = Series([np.median(nr.choice(probs, size=nsamp, replace=True)) for _ in range(nsims)])
    return meds


# In[ ]:

def bootstrap_wd_prob_slow(probs: Series, nsims=100, nsamp=1000):
    meds = Series([probs.sample(nsims, replace=True, random_state=None).median() for _ in range(nsamp)])
    return meds


# In[ ]:

get_ipython().magic('time meds = bootstrap_wd_prob(gdf.Val)')


# In[ ]:

uprob_med_plot = uprob_books.assign(Book=lambda x: x.Book - 1).groupby('Book').Val.median()
del uprob_med_plot


# In[ ]:

len(meds), len(meds2)


# In[ ]:

prob_books.groupby('Book').size()


# In[ ]:

simfunc = booker(partial(bootstrap_wd_prob, nsims=1000, nsamp=1000))
sims = pd.concat([simfunc(gdf.Val, bk) for bk, gdf in uprob_books.groupby('Book')]).reset_index(drop=1)


# In[ ]:

simfunc = booker(partial(bootstrap_wd_prob, nsims=1000, nsamp=2000))
sims2k = pd.concat([simfunc(gdf.Val, bk) for bk, gdf in uprob_books.groupby('Book')]).reset_index(drop=1)


# In[ ]:

simfunc = booker(partial(bootstrap_wd_prob, nsims=1000, nsamp=10000))
sims1k10k = pd.concat([simfunc(gdf.Val, bk) for bk, gdf in uprob_books.groupby('Book')]).reset_index(drop=1)


# In[ ]:

simfunc = booker(partial(bootstrap_wd_prob, nsims=10000, nsamp=1000))
sims10k1k = pd.concat([simfunc(gdf.Val, bk) for bk, gdf in uprob_books.groupby('Book')]).reset_index(drop=1)


# In[ ]:

plt.figure(figsize=(16, 10))
plt.title('1000 sims, 10000 samps')
sns.violinplot(x='Book', y='Val', data=sims10k1k);


# In[ ]:

plt.figure(figsize=(16, 10))
plt.title('1000 sims, 10000 samps')
sns.violinplot(x='Book', y='Val', data=sims2k);


# In[ ]:

uprob_books.groupby('Book').Val.median()


# In[ ]:

plt.figure(figsize=(16, 10))
plt.title('1000 sims, 1000 samps')
sns.violinplot(x='Book', y='Val', data=sims);
plt.xlim(-.5, 6.5);


# In[ ]:

plt.figure(figsize=(16, 10))
plt.title('200 sims, 1000 samps')
sns.violinplot(x='Book', y='Val', data=sims2);


# In[ ]:

plt.figure(figsize=(16, 10))
plt.title('100 sims, 1000 samps')
sns.violinplot(x='Book', y='Val', data=sims);


# In[ ]:

sims


# In[ ]:

meds.hist(alpha=.5)


# In[ ]:




# In[ ]:

nsims = 100


# In[ ]:

meds = Series([gdf.Val.sample(nsims, replace=nsamp, random_state=None).median() for _ in range(nsamp)])


# In[ ]:

gdf[:2]


# In[ ]:

len(gdf)


# In[ ]:

uprob_books[:2]


# In[ ]:




# In[ ]:




# In[ ]:

def get_diff(p1, p2, justdiff=False):
    pct1 = np.percentile(p1, 5)
    pct2 = np.percentile(p2, 5)
    if justdiff:
        return pct2 - pct1
    return pct1, pct2, pct2 - pct1

print('Actual 95%-tile rarest words: Book 1: {:.2f}, Book 2: {:.2f}, Diff: {:.4f}'.format(*get_diff(probs1, probs2)))


# In[ ]:

all_sims = []


# In[ ]:

def simulate_pctile_diff(probs12, probs1, nsim=200):
    probs12 = np.array(probs12)
    N = len(probs12)
    l1 = len(probs1)
    l2 = N - l1
    p = l1 / N
    nr.seed(0)
    
    def shuffle_groups3():
        g1 = nr.binomial(1, p , N)
        p1 = probs12[g1 == 1]
        p2 = probs12[g1 == 0]
        return get_diff(p1, p2)
    
    for _ in range(nsim):
        all_sims.append(shuffle_groups3())
#     return Series([shuffle_groups3() for _ in range(nsim)])


# In[ ]:

get_ipython().magic('time simulate_pctile_diff(probs12, probs1, nsim=20000)')


# In[ ]:

filt = lambda x: ((x > -18.5) & (x < -18))


# In[ ]:

_prob = DataFrame([(tok.orth_, tok.prob) for tok in bktks[2] if filt(tok.prob)], columns=['Word', 'Prob'])
_prob.Prob = _prob.Prob.round(3)
_prob[filt(_prob.Prob)]
_prob.query('Prob == [-18.317, -18.257]')


# In[ ]:

s2[filt(s2)].value_counts(normalize=0)


# In[ ]:




# In[ ]:

all_sims


# In[ ]:

sall_sims = Series(all_sims)
(sall_sims < get_diff(probs1, probs2)[2]).mean()


# In[ ]:




# In[ ]:

sall_sims.hist(bins=100, normed=1)


# In[ ]:

get_ipython().magic('time sims1 = simulate_pctile_diff(probs12, probs1, nsim=60000)')


# In[ ]:

get_ipython().magic('time sims = simulate_pctile_diff(probs12, probs1, nsim=200)')


# In[ ]:

sims2


# In[ ]:

get_ipython().magic('time sims2 = simulate_pctile_diff(probs12, probs1, nsim=2000)')


# In[ ]:

sims2.hist(bins=50, normed=1)
sims1.hist(bins=50, normed=1)


# In[ ]:




# In[ ]:




# In[ ]:

nr.ber


# In[ ]:

l1, l2


# In[ ]:

nr.shuffle()


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

