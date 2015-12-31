
# coding: utf-8

# In[ ]:

from py3k_fix import *


# In[ ]:

get_ipython().run_cell_magic('javascript', '', "IPython.keyboard_manager.command_shortcuts.add_shortcut('Ctrl-k','ipython.move-selected-cell-up')\nIPython.keyboard_manager.command_shortcuts.add_shortcut('Ctrl-j','ipython.move-selected-cell-down')\nIPython.keyboard_manager.command_shortcuts.add_shortcut('Shift-m','ipython.merge-selected-cell-with-cell-after')")


# In[ ]:

from collections import namedtuple, Counter, defaultdict, OrderedDict
from glob import glob
import operator as op
from os.path import join
import re
import toolz.curried as z

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
        self.chapters = chapters
        self.title = title
        self.txts = OrderedDict()
        for n, chap in sorted(chapters.items()):
            setattr(self, 't{}'.format(n), chap.text)
            self.txts[n] = chap.text
        txt = reduce(op.add, self.txts.values())
        self.txt = clean_text(txt)


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


# In[ ]:

n = 7
bks = {i: parsebook(i, vb=1) for i in range(1, n + 1)}


# In[ ]:

bookpat.search(txt)


# In[ ]:

txt[:2000]


# In[ ]:

print(txt[:2000])


# In[ ]:

bks


# In[ ]:

bookpat = re.compile(r'''\A(?P<title>.+)
(?!(.+)+?)
(?P<body>(Chapter 1)\n+(.+\n+)+)''')
chappat = re.compile(r'''(Chapter (\d+)\n+((?:.+\n+)+))+''')
chapsep = re.compile(r'Chapter (\d+)\n(.+)\n+')
# m = bookpat.search(t)
# gd = m.groupdict()
# title = gd['title']
# body = gd['body']
# gd


# book = {int(chnum): Chapter(int(chnum), title, text) for chnum, title, text in z.partition(3, chs)}
# bk = Book(book)
# book

# In[ ]:




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

bk = parsebook()


# In[ ]:

print(bk.txt[:2000])


# In[ ]:

Counter(filter(lambda x: not re.match(r'[\dA-Za-z \."\'\-\(\);:!\?,\n]', x), bk.txt))


# In[ ]:

Complexity
- avg word length (syllables?)
- avg sentence length

- Words: 129 | Syllables: 173 | Sentences: 7 | Characters: 568 | Adverbs: 4
Characters/Word: 4.4 | Words/Sentence: 18.4


# # Text

# In[ ]:




# In[ ]:

import spacy.parts_of_speech as ps
import spacy.en

nlp = spacy.en.English()


# In[ ]:

tokens = nlp(bk.txt, tag=True, parse=False, entity=True)


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

