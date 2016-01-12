
# coding: utf-8

# I finally started reading a series of books over break featuring a wizard named Harry Potter. As I progressed through the series, I noticed the page count getting longer and the books getting heavier. While recounting this to a friend who had also read the series, she noted that she felt the writing style of the books seemed to grow as much as the thickness. While the earlier books (particularly the first) seemed to be written to a younger audience with simpler sentences and descriptions, she said, this seemed to get more complex in the later books as Harry also comes of age. I have heard this sentiment confirmed with other readers, and began to wonder whether it would be possible to quantify writing complexity and measure whether there is a significant difference as the series progress. This [notebook](somewhere) is my attempt to quantify it.  
# 
# To quantify the complexity, I look at the following metrics: average word length and sentence length, word frequency within the text, word frequency based on external sources, and the average syntactic complexity of sentences.
# 
# The testing to determine the difference between books is largely inspired by seeing the recent presentation ["Statistics for Hackers"](https://speakerdeck.com/jakevdp/statistics-for-hackers) by Jake VanderPlas, where he introduces simulations, label shuffling, bootstrapping and cross validation as intuitive alternatives to classical statistical approaches. The two methods that I mainly use here are permutation testing ('label shuffling') and the bootstrap. I also used this as a chance to try out the [spaCy](https://spacy.io) library, which bills itself as a "library for industrial-strength natural language processing in Python and Cython."
# 
# From here on out, the words will be much sparser than the code.

# # Import, load/clean/parse text
# See `utils.py` for cleaning and parsing details.

# In[ ]:

get_ipython().run_cell_magic('javascript', '', "var csc = IPython.keyboard_manager.command_shortcuts\ncsc.add_shortcut('Ctrl-k','ipython.move-selected-cell-up')\ncsc.add_shortcut('Ctrl-j','ipython.move-selected-cell-down')\ncsc.add_shortcut('Shift-m','ipython.merge-selected-cell-with-cell-after')")


# In[ ]:

from project_imports import *
import utils as ut; reload(ut);
get_ipython().magic('matplotlib inline')

cachedir = 'cache/'
memory = Memory(cachedir=cachedir, verbose=0)


# In[ ]:

bksall = ut.BookSeries(7)

with open('src/stops.txt', 'r') as f:
    stops = set(l for l in f.read().splitlines() if l and not l.startswith('#'))


# The starting point for spaCy is a parsing function that parses, tags and detects entities all in one go.

# In[ ]:

from spacy.en import English
from spacy.parts_of_speech import ADJ
get_ipython().magic('time nlp = English()')


# In[ ]:

# bktks = {i: nlp(bktxt, tag=True, parse=True, entity=True) for i, bktxt in bks.txts.items()}
get_ipython().magic('time bktksall = {i: nlp(bktxt, tag=True, parse=True, entity=True) for i, bktxt in bksall.txts.items()}')


# I'll be writing a bunch of functions that take a list of tokens and returns a list of processed strings, numbers, etc. The following higher order functions are to facilitate applying these `[Token] -> [a]` functions to the entire Harry Potter series, returning a dataframe that keeps track of which book the processed value in a given row came from.

# In[ ]:

def tobooks(f: '(toks, int) -> DataFrame', bks=bktksall) -> DataFrame:
    """Apply a function `f` to all the tokens in each book,
    putting the results into a DataFrame column, and adding
    a column to indicate each book.
    """
    return pd.concat([f(v, i) for i, v in bks.items()])

def booker(f: 'toks -> [str]') -> '(toks, int) -> DataFrame':
    @wraps(f)
    def tobookdf(toks, bknum):
        res = f(toks)
        if np.ndim(res) == 1:
            df = DataFrame(f(toks), columns=['Val'])
        else:
            df = res
        df['Book'] = bknum
        return df
    return tobookdf
    
over_books = z.comp(partial(tobooks, bks=bktksall), booker)


# As an example, a function that takes a stream of tokens and returns the first 2 words that are adjectives,

# In[ ]:

fst_2_nouns = lambda xs: list(it.islice((x.orth_ for x in xs if x.pos == ADJ), 2))


# can be applied to each book in the series as:

# In[ ]:

over_books(fst_2_nouns)


# # Search for increasing complexity
# ## Average word and sentence length
# 
# A first simple search would be to see if the average length of the words or sentences increases throughout the series.

# In[ ]:

spanlen = lambda span: len([wd for wd in span if len(wd) > 1])
sent_lensf = lambda parsed: [spanlen(sent) for sent in parsed.sents]
wd_lensf = lambda parsed: [len(tok) for tok in parsed if len(tok) > 1]


# In[ ]:

wd_lensb = over_books(wd_lensf)
sent_lensb = over_books(sent_lensf)


# In[ ]:

def wd_sent_lens():
    def agg_lens(lns):
        return (lns
              .groupby('Book')['Val'].agg(['mean', 'median', 'std'])
              .rename(columns=str.capitalize))

    wd_len = agg_lens(wd_lensb)
    sent_len = agg_lens(sent_lensb)
    
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


# There does appear to be an increasing trend for both average word and sentence length difference between the books, though the scale of the difference looks miniscule in light of the standard deviations within each book.
# 
# One way to gauge the likelihood that the average word length difference between, say, books 1 and 2 is due to chance would be to   shuffle the labels a bunch of times, and each time calculate the difference in average word length. (See Jake VanderPlas' [Statistics for Hackers](https://speakerdeck.com/jakevdp/statistics-for-hackers) talk and [Tim Hesterberg's article](http://arxiv.org/abs/1411.5279) covering permutation tests for an explanation).

# In[ ]:

def sim_diff_(xs, N, aggfunc=np.mean, p=.5):
    labs = nr.binomial(1, p, N)  # cheap shuffling approximation
    g1 = aggfunc(xs[labs == 1])
    g2 = aggfunc(xs[labs == 0])
    return g2 - g1


@memory.cache
def sim_diff(xs, n1, aggfunc=np.mean, nsims=10, n_jobs=1):
    N = len(xs)
    p = n1 / N
    xs = np.array(xs)
    f = delayed(sim_diff_)
    gen = (f(xs, N, aggfunc=aggfunc, p=p) for _ in range(nsims))
    return Series(Parallel(n_jobs=n_jobs)(gen))


def plot_perm_diffs(samps, actual=None, bka=2, bkb=1, subplt=1, xlabel=None):
    t = ('Simulated and actual difference between books {bka} and {bkb}'
         '\nPermutation pvalue: {pv:.3%}; N={N:,.0f}'
          .format(bka=bka, bkb=bkb, pv=ut.pvalue(actual, samps), N=len(samps)))
    plt.subplot(1, 2, subplt, title=t)
    samps.hist(bins=50)
    plt.vlines(actual, *plt.ylim())
    plt.legend(['Actual\ndifference'], loc=2)
    if xlabel is not None:
        plt.xlabel(xlabel)


# In[ ]:

wd_lensb23 = wd_lensb.query('Book == [2, 3]')
get_ipython().magic('time perm_wd_23 = sim_diff(wd_lensb23.Val, wd_lensb23.Book.value_counts(normalize=0)[2], nsims=10000, n_jobs=-1)')


# In[ ]:

wd_lensb12 = wd_lensb.query('Book == [1, 2]')
get_ipython().magic('time perm_wd_12 = sim_diff(wd_lensb12.Val, wd_lensb12.Book.value_counts(normalize=0)[1], nsims=10000, n_jobs=-1)')


# As the following histogram of simulated word length differences shows, the difference is quite significant in the word lengths, despite what the large standard deviations first led me to believe. The trendline above shows an pronounced jump between books 1 & 2 which is reflected in the smallest possible p-value, but the permutation sampling shows that the jump between 2 and 3 is also significant.

# In[ ]:

plt.figure(figsize=(16, 5))
dw12 = wsls.Word_length.Mean[2] - wsls.Word_length.Mean[1]
plot_perm_diffs(perm_wd_12, actual=dw12, bka=1, bkb=2, subplt=1, xlabel='Word length difference')

dw23 = wsls.Word_length.Mean[3] - wsls.Word_length.Mean[2]
plot_perm_diffs(perm_wd_23, actual=dw23, bka=2, bkb=3, subplt=2, xlabel='Word length difference')


# The earlier trendline for sentence lengths is more ambiguous. While the word length immediately jumps following *The Sorcerer's Stone* and continues to increase in all but 2 cases, the sentence length bounces around a lot more. But there does seem to be a significant difference between the first four and the last three, which we can also test for.

# In[ ]:

sent_lens_fst_snd = sent_lensb.copy()
sent_lens_fst_snd['Part'] = '1-4'
sent_lens_fst_snd.loc[sent_lensb.Book > 4, 'Part'] = '5-7'
μ_ab = sent_lens_fst_snd.groupby('Part').Val.mean()

n1 = sent_lens_fst_snd.Part.value_counts(normalize=0)['1-4']
get_ipython().magic('time perm_sent_ab = sim_diff(sent_lens_fst_snd.Val, n1, nsims=10000, n_jobs=-1)')
del n1
sent_lens12 = sent_lensb.query('Book == [1, 2]')
get_ipython().magic('time perm_sent_12 = sim_diff(sent_lens12.Val, sent_lens12.Book.value_counts(normalize=0)[1], nsims=100000, n_jobs=-1)')


# In[ ]:

plt.figure(figsize=(16, 5))
dsab = μ_ab.ix['5-7'] - μ_ab.ix['1-4']
plot_perm_diffs(perm_sent_ab, actual=dsab, bka='5-7', bkb='1-4', subplt=1, xlabel='Sentence length difference')

ds12 = wsls.Sentence_length.Mean[2] - wsls.Sentence_length.Mean[1]
plot_perm_diffs(perm_sent_12, actual=ds12, bka=1, bkb=2, subplt=2, xlabel='Sentence length difference')


# Here we see that the difference between books 1 & 2 is very significant, though not as much as the difference between the first and second parts (as defined by 1-4 and 5-7). Thus we can, with reasonable confidence, reject the notion that the increase in word length and sentence length through the series is a statistical artifact that ocurred by chance. 

# ## Word complexity by frequency
# 
# In addition to measuring the increase in average word/sentence length, it could be more insightful to measure the frequency of rarer words. My hypothesis was that even complex language is still largely composed of shorter, commoner words, highlighted by rarer, more complex words. A way to test this could be to somehow get a measure of the frequency of just the rarer words by counting, for example, what percentage of the words only appear once.

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


# ## Word complexity by English language frequency

# The frequency counting above, however, only counts words that are rare within the context of this series. Fortunately, spaCy provides a log-probability score for each parsed word, based on its frequency in external corpora. These will be negative numbers such that a lower score indicates that a word is less common in English usage outside of Harry Potter. "Low probability," "low likelihood" and "less common" are terms I'll use to describe words with low log-probability scores.

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


# The most drastic difference is in the frequency of the 95th percentile between first and second books. The graph shows that a typical word in the 95th percentile has a log probability of -13.3 in the first book and -13.8 in the second (this is about the difference between the words *frogs* and *lopsided*). The drop doesn't look that drastic, and there doesn't seem to be a discernable overall trend, either.
# 
# Out of curiosity, it could be helpful to dig into what the probabilities look like for the first couple hundred least likely words.

# In[ ]:

probs1 = probs(bktksall[1])
probs2 = probs(bktksall[2])
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
plt.hlines([-18.25, -18.32], *plt.xlim(), linestyles='dashdot');


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


# Indeed, the relationship between typical word probability appears to have a quite [log] linear relationship with word count. I'm not sure what relationship is to be expected, but it looks like it would be worthwhile to try and correct for document length in determining word complexity.
# 
# Before digging into this correction, here is what the growth pattern looks like for unique words, sampling and shuffling several times. The cumulative count of new words resembles a log curve, and the new word rate resembles an exponential decay curve (at least to my eyes).

# In[ ]:

def simgrowth(toks, nsims=20, window=200):
    def simgrowth_():
        s = set()
        l = []
        tks = map(prop('orth'), toks)
        nr.shuffle(tks)
        for w in tks:
            s.add(w)
            l.append(len(s))
        sl = Series(l)
        nl = sl.values
        rate = Series(pd.rolling_mean(nl[1:] - nl[:-1], window)).dropna()
        return sl, rate
    
    return [simgrowth_() for _ in range(nsims)]

ls = simgrowth(bktksall[1], nsims=10)


# In[ ]:

plt.figure(figsize=(16, 5))
for l, rm in ls:
    plt.subplot(1, 2, 1)
    plt.title('New word')
    plt.plot(l, alpha=.1, )
    
    plt.subplot(1, 2, 2)
    plt.title('New word rate')
    rm.plot(alpha=.1)
    
μ_rm = pd.concat(map(itg(1), ls), axis=1).mean(axis=1)
μ_rm.plot(alpha=1)


# ### Probability of unique words: correcting for varying word counts
# ####Simulate word distributions
# For each booklength $L$, I'll be repeatedly sampling $L$ words without replacement from the book with the largest word count, book 5, and then finding the average unique word probability of each sample. This should give an estimate of what the average word count should be for each book, they were all drawing from the same source, given the length of each book. 

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


# In[ ]:

@memory.cache
def get_gen_prob_text(nsims=10000, n_jobs=-1, worddist=5, rep=False):
    gens_mus = {
        booknum: sim_gen_text(worddist=worddist, sizebook=booknum,
                              nsims=nsims, aggfunc=np.mean, n_jobs=n_jobs, rep=rep)
        for booknum in range(1, 8)
    }
    d = DataFrame(gens_mus)
    return d

def join_sim_act(simdf_):
    cols = ['Val', 'Book', 'Source']
    simdf = simdf_.copy()
    simdf.columns.name = 'Book'
    simdf = simdf.stack().sort_index(level='Book').reset_index(drop=0).rename(columns={0: 'Val'}).drop('level_0', axis=1)
    simdf['Source'] = 'Simulation'
    dboth = simdf[cols].append(uprob_books.assign(Source='Actual')[cols]).sort_values(['Book', 'Source'], ascending=True)
    return dboth, simdf


# In[ ]:

get_ipython().magic('time simdf_ = get_gen_prob_text(nsims=10000, worddist=8, n_jobs=-1, rep=False)')
dboth, simdf = join_sim_act(simdf_)
bothagg = dboth.groupby(['Source', 'Book',]).mean()
bothagg.unstack('Source')


# In[ ]:

plt.figure(figsize=(16, 10))
pbothagg = ut.mod_axis(bothagg.ix['Actual'].copy().rename(columns={'Val': 'Actual'}),
                       z.operator.add(-1))
plt.scatter([], [], s=80, c='k', marker='x', linewidth=2)
plt.legend(['Actual']);
sns.violinplot('Book', 'Val', data=simdf)
plt.scatter(pbothagg.index, pbothagg.Actual, s=80, c='k', marker='x', linewidth=2);


# Barring some subtle errors in my simulation code (which would not surprise me at all), the violin plot above says that the actual average word probability for books 2, 5, 6 and 7 are roughly what one would expect if words were drawn at random from the whole series, based solely on the length of the book. Measuring word complexity as having a low probability, this could lead one to say that the word complexity of the first book is way below average, as well as the 3rd and 4th to a lesser extent, with 5, 6 and 7 increasingly approaching the average. This seems to be the best evidence so far of the writing complexity increasing as Harry Potter's education progresses.
# 
# The trend in increasing complexity is perhaps clearer when plotting this difference in simulated and actual average probability:

# In[ ]:

plt.figure(figsize=(16, 5))
plt.subplot(1, 2, 1)
bothagg.unstack('Source')['Val'].eval('Simulation - Actual').plot(title='Average - actual word complexity');
plt.subplot(1, 2, 2)
plot_corrcoef(x='Word_count', y='Mean', data=simdf.groupby(['Book']).mean().rename(columns={'Val': 'Mean'}).assign(Word_count=wc))


# To the right, we also see that at least the simulated values are much better estimated by a linear word count predictor (negative correlation coefficient of .985 for the simulated vs .935 for the actual averages plotted earlier). 

# ## Sentence structure complexity

# There are different ways to measure the complexity of a sentence based on the syntactical structure, 

# In[ ]:

# import networkx as nx
import pygraphviz as pgv


# In[ ]:

def build_graph(s, add_edge=ut.add_edge):
    G = pgv.AGraph(directed=True)
    reprdct = ut.dedupe_wrd_repr(s)
    
    def build_graph_(tok, i=0):
        for c in tok.children:
            add_edge(tok, c, G, reprdct=reprdct)
            build_graph_(c, i=i + 2)
        return G
    return build_graph_(s.root)


def show_graph(g):
    g.draw("file.png", prog='dot')
    return Image(filename="file.png") 


s = next(bktksall[1].sents)
G = build_graph(s)
Gi = build_graph(s, add_edge=ut.add_int_edge)


# In[ ]:

show_graph(G)


# The two features to compare across books that came to mind looking at this structure are the total height of the whole tree, and the depth of the nodes, averaged for each sentence. While I used permutation tests to analyze the other metrics above, here I'll run bootstrap simulations of the syntax tree depths, and average the difference between books over the simulated values, just to mix things up.

# In[ ]:

def tree_depths(s, senti=None):
    def tree_depths_(tok, i=1, vb=False,):
        return [(i, senti)] + [t for c in tok.children
                      for t in tree_depths_(c, i=i + 1, vb=vb)]
    return tree_depths_(s.root, i=1)


def sent_depth_bk(toks):
    return DataFrame([(depth, i) for i, s in 
                      enumerate(toks.sents)
                      for depth, senti in tree_depths(s, senti=i)],
                    columns=['Depth', 'Sentnum'])


sent_depths = over_books(sent_depth_bk).reset_index(drop=1)


# In[ ]:

sgb = (sent_depths.groupby(['Book']).Depth
       .agg(['mean', 'median', 'max', 'idxmax'])
       .rename(columns=str.capitalize))
sgb


# In[ ]:

def sim_depth(s, seed=None, size=None, aggfunc=None):
    return aggfunc(s.sample(n=size, replace=True, random_state=seed))


@memory.cache
def bootstrap_depths(df, by='Book', col=None, aggfunc=np.mean,
                     nsims=10, size=1000, n_jobs=1):
    genmkr = lambda s: (delayed(sim_depth)(s, seed=seed, aggfunc=aggfunc, size=size) for seed in range(nsims))
    df = DataFrame({bknum: Parallel(n_jobs=n_jobs)(genmkr(gbs)) for bknum, gbs in df.groupby(by)[col]} )
    return df

def piv(df):
    df.columns.name = by
    return (df.unstack().reset_index(drop=0).drop('level_1', axis=1)
            .rename(columns={0: 'Val'}))


# In[ ]:

get_ipython().magic("time bootdepths = bootstrap_depths(sent_depths, by='Book', col='Depth', nsims=10000, n_jobs=-1)")


# Here are the simulated average depths of each word by book for comparison:

# In[ ]:

ut.mod_axis(sgb, z.operator.add(-1)).Mean.plot()
sns.violinplot(data=bootdepths);


# A difference of 0 doesn't overlap much (if at all) with the distribution of the bootstrapped samples, giving us reason to believe that the difference in syntactical complexity is significant at least between books 1 and 5. This contrasts with the difference between books 1 and 2--while the the average difference is about .05 levels, the simulations make a hypothesis of 0 difference look plausible:

# In[ ]:

def plot_bt_diffs(samps, bka, bkb, subplt=1):
    diff = samps[bka] - samps[bkb]
    t51 = ('Average depth: Book {bka} - book {bkb} \n(0 > difference in {perc:.2%}'
           ' of examples)'.format(bka=bka, bkb=bkb, perc=(0 > diff).mean()))
    plt.subplot(1, 2, subplt, title=t51)
    diff.hist(bins=50)
    plt.vlines(0, *plt.ylim())


# In[ ]:

plt.figure(figsize=(16, 5))
plot_bt_diffs(bootdepths, 2, 1, subplt=1)
plot_bt_diffs(bootdepths, 5, 1, subplt=2)


# ### Height

# One more metric I would like to look at is the average 'height' of the sentences by books. While I previously looked at the average depth of each word in the syntax tree, this will just talky the maximum depth of each sentence.

# In[ ]:

maxdepth = (sent_depths.groupby(['Book', 'Sentnum']).Depth.max()
        .reset_index(drop=0))
sgbs = (maxdepth.groupby(['Book']).Depth
        .agg(['mean', 'median', 'max'])
        .rename(columns=str.capitalize)
       )
sgbs


# Here again we see a bit of variation in the average sentence height by each book, but it's not obvious whether these differences are significant. Time for the bootstrap again!

# In[ ]:

get_ipython().magic("time bootheights = bootstrap_depths(maxdepth, by='Book', col='Depth', nsims=100000, n_jobs=-1, size=1000)")


# In[ ]:

plt.figure(figsize=(16, 5))
plot_bt_diffs(bootheights, 2, 1, subplt=1)
plot_bt_diffs(bootheights, 5, 1, subplt=2)


# In the case of measuring the difference in average sentence *heights* between the books, we have much more confidence that the difference in books 5 and 1 *and* between 2 and 1 were not due to chance.

# In[ ]:

sns.violinplot(data=bootheights);


# # Conclusion

# ## Epilogue: fun stats

# #### Longest words

# In[ ]:

Series([tok.orth_ for bk in bktksall.values() for tok in bk if len(tok.orth_) > 20]).value_counts(normalize=0)


# #### Longest unhyphenated words

# In[ ]:

Series([tok.orth_ for bk in bktksall.values()
        for tok in bk if len(tok.orth_) > 15
        and '-' not in tok.orth_]).value_counts(normalize=0)


# #### Tallest sentence in the series

# In[ ]:

_, maxsentnum, maxbooknum = sent_depths.ix[sent_depths.Depth.idxmax()]
[sent] = list(it.islice(bktksall[maxbooknum].sents, int(maxsentnum), int(maxsentnum + 1)))
print(sent)
show_graph(build_graph(sent))


# #### Longest sentences
# These seem to show some parsing issues, where a sequence of quick dialogue or lyrics are interpreted as a single sentence

# In[ ]:

get_ipython().system('date')


# In[ ]:

get_ipython().system('date')


# In[ ]:

for s in [sent for _, bk in sorted(bktksall.items())[:5] for sent in bk.sents if spanlen(sent) > 200]:
    print(s, end='\n=======\n')


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

sent_lens = pd.concat([sent_lens(v, i) for i, v in bktks.items()])


# In[ ]:

# Longest sentences
Series().value_counts(normalize=0)


# In[ ]:




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

people = Series(ent.string.rstrip() for ent in tokens.ents if ent.label_ == 'PERSON')


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

