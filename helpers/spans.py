import numpy as np
import pandas as pd
import spacy
from spacy import displacy
from spacy.tokens import Span

def spans_start(df):
    if len(df)==1:
        return df
    start = df.start.unique()[0]
    end = np.max(df.values)
    return pd.DataFrame([{"start":start,"end":end}])

def spans_end(df):
    if len(df)==1:
        return df
    end = df.end.unique()[0]
    start = np.min(df.values)
    return pd.DataFrame([{"start":start,"end":end}])

def spanss(df):
    df= df[['start','end']].groupby('start').apply(spans_start)
    df_start=df.reset_index(drop=True)
    df_end = df_start.groupby('end').apply(spans_end).reset_index(drop=True)
    return df_end.reset_index(drop=True)

def pep_matches_top_1(list_of_matches):
    answers={}
    if type(list_of_matches)!=list:
        list_of_matches = eval(list_of_matches)
    for ans in list_of_matches:
        name = list(ans.keys())[0]
        pep_list = list(ans.values())
        if not pep_list==[{}]:
            pep_name = list(sorted(pep_list[0].items(), key=lambda x: x[1], reverse=True)[0])[0]
            answers.update({name:pep_name})
    return answers

import re
def tmp_fn(df):
    # print(df)
    if type(df)==tuple:
        df=df[1]
        df = df.reset_index()
    text = list(set(df['text'].tolist()))[0]
    lst_ents=[]
    nlp = spacy.blank("en")
    doc = nlp(text)
    options = {"ents": ["PEP","Reason"], "colors":  {"PEP": "red", "Reason":"yellow"}}
    # options = {"ents": ["PEP"], "colors":  {"PEP": "red"}}
    for i, row in df.iterrows():
        accuse_char = doc.char_span(row['start_why'], row['end_why'])
        if row['pred_ans']=={}:
            continue
        for pep in row['pred_ans']:
            who_char = re.search(pep, text).span()
            who_char=doc.char_span(who_char[0], who_char[1])
            span = Span(doc, who_char.start, who_char.end, "PEP")
            reason = Span(doc, accuse_char.start, accuse_char.end, "Reason")
            if span not in lst_ents:
                lst_ents.extend([
                    span,
                ])
            if reason not in lst_ents:
                lst_ents.extend([
                    reason,
                ])
    doc.ents = lst_ents
        
    return doc