
import spacy_streamlit

import streamlit as st

import numpy as np
import pandas as pd
# import nltk
# import tqdm
# from transformers import pipeline
# from torch.utils.data import Dataset, DataLoader

from helpers.spans import pep_matches_top_1, spanss, tmp_fn
from qa import ask_question, get_multi_ner, check_pep, why_pep
# from ner import get_ner, get_named_entities, name_match, name_match_single

# model_name = "mirbostani/bert-base-uncased-finetuned-newsqa"
# qa_model = pipeline(model=model_name, tokenizer=model_name, task="question-answering")    

DEFAULT_TEXT = "Six bankers record statements against Shehbaz, Hamza in money laundering case. Shehbaz Sharif and his son Hamza Shahbaz.The bankers have recorded their statement under section 164 of the Code of Criminal Procedure in an Rs25 billion money laundering case registered by the Federal Investigation Agency (FIA) against the father and son duo.According to sources, the FIA has presented a list of 26 bankers alleging that the Shehbaz and Hamza opened various bank accounts in the name of security guards, watchmen, clerks working in the Ramzan Sugar Mills.Special Judicial Magistrate Muhammad Kamran Zafar had summoned the witnesses to record their statements today. He had also summoned Shehbaz and Hamza to appear for the cross-examination but were a no show.The magistrate after recording statements of six bankers adjourned the proceedings of the case for September 23 to record statements of the rest of the 20 witnesses. \nAmong the 26 witnesses to record statements against Shahbaz Sharif and Hamza Shahbaz include Maleeha Yousaf of HBL New Muslim Town branch Lahore, Iram Batool Hashmi of MCB Bahawalpur branch, Afzal Ghauri Bhatti of MCB Karim Block, Iqbal Town branch Lahore, Ishfaq Ahmad of MCB Faisalabad branch, Hamid Ali of MCB Mianwali branch, Tayyab Ali of MCB Jhang Road branch, Chiniot, Muhammad Atif Ameer of MCB Sargodha branch, Afzaal Ahmad of MCB Akbar Chowk branch Lahore, Asif Naeem of MCB Jamia Bad branch, Syed Tabbassum Raza Naqvi of MCB Jamia Bad branch, Ahmad Ali of MCB Jhang Road branch, Chiniot, Khurram Shahzad of MCB Chenab Nagar branch, Chiniot, Muhammad Rehman Sarwar of MCB Faisalabad, Fareeha Ijaz of MCB Ajmal House branch, Lahore, Naveed Wahab of UBL, Chiniot branch, Tanvir Hussain of Meezan Bank, Chiniot branch, Asif Iqbal of UBL Chenab Nagar branch, Chiniot, Farrukh Hussain of Meezan Bank of Adda branch, Sargodha, Ghulam Mustafa of Allied Bank main branch Chiniot, Nadeem Ahmad, Kotwali branch Faisalabad, Naseer Ahmad of Allied Bank Faisalabad, Riaz Hussain of Allied Bank, Chiniot branch, Muhammad Zubair of Meezan Bank Lalian branch, Chiniot, Shakeel Ahmad of UBL Sargodha Road branch, Chiniot and Muhammad Hashaam of Meezan Bank, Main branch, Chiniot. \nIt may be relevant to mention here that following investigations into the sugar scam, FIA Lahore had on November 14, 2020, registered an FIR No 39/20 against Shahbaz Sharif, Hamza Shahbaz and Salman Shahbaz for their alleged involvement in Rs 25 billion money laundering which FIA has claimed was committed through various benami accounts.Salman Shahbaz is currently in the UK and did not join the investigations whereas Shahbaz Sharif and Hamza Shahbaz have appeared before the FIA investigators couple of times and are currently on pre-arrest bails"

question = "What is the name of person being accused?"
question_why = "What is the reason <NAME> is being accused?"

st.title("PEP Identification")
text = st.text_area("Text to analyze", DEFAULT_TEXT, height=200)

results = ask_question(text, question, topK=1)
results['ner_results'] = results['answer'].astype(str).apply(get_multi_ner)
results['pep_match_results'] = results['ner_results'].apply(check_pep)
results_why = why_pep(results.answer.tolist(), results['text'].tolist(), question_why)
results['pred_ans'] = results['pep_match_results'].apply(lambda x: pep_matches_top_1(x))

df_spans = results_why.groupby('text').apply(spanss).reset_index()
df_merged = results.merge(df_spans,on='text', how='left', suffixes=('','_why'))
df_merged = df_merged.drop_duplicates(['start','end','start_why','end_why'])
doc = tmp_fn(df_merged)

spacy_streamlit.visualize_ner(
    doc,
    labels=["PEP", "Reason"],
    show_table=False,
    title="PEP and Reason for accustation",
    colors={"PEP": "red", "Reason":"yellow"}
)
# st.text(f"Analyzed using spaCy model {spacy_model}")