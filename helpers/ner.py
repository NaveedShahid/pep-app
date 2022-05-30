# from transformers import AutoTokenizer, AutoModelForTokenClassification
import pandas as pd 
import nltk
# import tqdm
import re
# import json
# from pprint import pprint
# from ipywidgets import HTML
# from IPython.core.display import display, HTML
import unicodedata
from thefuzz import fuzz
from flair.data import Sentence
from flair.models import SequenceTagger
# from flair.visual.ner_html import render_ner_html

pep_person_df = pd.read_csv('data/PEP name PERSON.csv', names=['names'])
tagger = SequenceTagger.load("flair/ner-english-ontonotes-large")

pep_list = pep_person_df.names.values.tolist()

def get_ner(paragraph, entity=None):
    ner_dict = {}
    for text in nltk.tokenize.sent_tokenize(paragraph):
        text = re.sub('\u2019s','',text)
        text = unicodedata.normalize('NFKD',  text).encode('ascii', 'ignore').decode("UTF-8")
        text = text.encode("ascii", "ignore").decode('UTF-8')

        # text= re.sub('[\n]+','. ', text)
        # text= re.sub('[\W+]', ' ', text)
        sentence = Sentence(text)
        tagger.predict(sentence)
        for ent in sentence.get_spans('ner'):
            ner_type = ent.annotation_layers['ner'][0]._value

            if entity==ner_type:
                ent_list = [e.text for e in ent.tokens]
                if len(ent_list)>1:
                    ner_value = ' '.join(ent_list)
                    ner_dict.update({ner_value:ner_type})
            elif entity is None:
                ner_value = ' '.join(e.text for e in ent.tokens)
                ner_dict.update({ner_value:ner_type})                

    return ner_dict

def get_named_entities(text):
    if len(text) > 10:
        named_entities = [list(get_ner(t,'PERSON').keys()) for t in nltk.tokenize.sent_tokenize(text)]
    else:
        return []
    
    named_entities = sum(named_entities,[])

    named_entities = list(set(named_entities))
    for n in named_entities:
        for m in named_entities:
            if m in n and m!=n:
                named_entities.remove(m)
    return named_entities

from itertools import combinations

def name_match_single(name, pep):
    name = re.sub('\W+',' ',name).strip().lower()
    pep = re.sub('\W+',' ',pep).strip().lower()

    pep = pep.lower()
    name_l = nltk.word_tokenize(name)
    pep_l = nltk.word_tokenize(pep)
    if name.lower()==pep.lower():
        return 1.0

    ratio_full = fuzz.ratio(name.lower(),pep.lower())
    if ratio_full>=90:
        return ratio_full/100

    name_all_combs =[l for i in range(len(name_l)) for l in combinations(name_l, i+1)]
    pep_all_combs =[l for i in range(len(pep_l)) for l in combinations(pep_l, i+1)]

    if len(name_l)<len(pep_l):
        n= len(pep_l)-1
        if len(name_l)>=n and n>=2:
            name_n_1 = list(filter(lambda x: len(x)==n, name_all_combs))
            pep_n_1 = list(filter(lambda x: len(x)==n, pep_all_combs))
            name_n_1 = [' '.join(name_) for name_ in name_n_1]
            pep_n_1 = [' '.join(pep_) for pep_ in pep_n_1]
            for n in name_n_1:
                for n_1 in pep_n_1:
                    ratio_partial_1 = fuzz.ratio(n_1.lower(),n.lower())
                    if ratio_partial_1>=90:
                        return (ratio_partial_1/100) - 0.25

        elif len(name_l)>=n-1 and (n-1)>=2:
            name_n_2 = list(filter(lambda x: len(x)==n-1, name_all_combs))
            pep_n_2 = list(filter(lambda x: len(x)==n-1, pep_all_combs))
            name_n_2 = [' '.join(name_) for name_ in name_n_2]
            pep_n_2 = [' '.join(pep_) for pep_ in pep_n_2]
            for n in name_n_2:
                for n_2 in pep_n_2:
                    ratio_partial_2 = fuzz.ratio(n_2.lower(),n.lower())
                    if ratio_partial_2>=90:
                        return (ratio_partial_2/100) - 0.50
    else:
        # print('len')
        return 0.0
    return 0.0
def name_match(name, pep_list):
    matches_list={}
    for pep in pep_list:
        score =  name_match_single(name,pep)
        if score > 0:
            matches_list.update({pep:score})
    
    return {name:matches_list}

def get_multi_ner(text):
    lst = get_named_entities(text)
    if lst ==[]:
        return ''
    return ', '.join(lst)

def check_pep(answer):
    if type(answer)==float:
        return ''
    else:
        matches_dict=[]
        lst = answer.split(', ')
        for name in lst:
            matches = name_match(name, pep_list)
            matches_dict.append(matches)
        return matches_dict
