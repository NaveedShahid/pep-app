import numpy as np
import pandas as pd
# import nltk
# import tqdm
# import spacy
from transformers import pipeline
# from torch.utils.data import Dataset

from helpers.ner import get_ner, get_named_entities, name_match, check_pep, get_multi_ner, Sentence 

model_name = "mirbostani/bert-base-uncased-finetuned-newsqa"
qa_model = pipeline(model=model_name, tokenizer=model_name, task="question-answering")    

def ask_question(lst_of_text, question, topK=3):
    assert topK>0, 'TopK should be greater than 0.'
    """ Input: question and a list of text
        Outputs: Dataframe with topK answers per item in list"""
    # question_1 = "What is the name of person being accused?"
    if type(lst_of_text)==str:
        if len(lst_of_text)<20:
            raise Exception('Error length')
        lst_of_text = [lst_of_text]
    
    
    questions = [question]*len(lst_of_text)
    print(len(questions), len(lst_of_text))
    res = qa_model(question=questions, context=lst_of_text, topk=topK, batch_size=1)
    res= pd.DataFrame.from_records([res])
    res.index = list(range(1,len(res)+1))
    if topK==1:
        res['text'] = lst_of_text
    else:
        new_indexes = np.mean(np.array(res.index).reshape(-1, topK), axis=1)
        for i,n in enumerate(new_indexes):
            # res.loc[n-topK//2:n+topK//2,'text'] = i+1
            res.loc[n-topK//2:n+topK//2,'text'] = lst_of_text[i]
    res = res[['text','answer','score','start','end']]
    return res


def why_pep(names, lst_of_text, question, topK=1):
    assert topK>0, 'TopK should be greater than 0.'
    assert len(names)==len(lst_of_text), 'Length of name list and text list should be same'

    """ Input: question and a list of text
        Outputs: Dataframe with topK answers per item in list"""
    # question_1 = "What is the name of person being accused?"
    if type(lst_of_text)==str:
        if len(lst_of_text)<20:
            raise Exception('Error length')
        lst_of_text = list(lst_of_text)
    
    questions = [question.replace('<NAME>',name) for name in names]

    res = qa_model(question=questions, context=lst_of_text, topk=topK, batch_size=1)
    res= pd.DataFrame.from_records([res])
    res.index = list(range(1,len(res)+1))
    if topK==1:
        res['text'] = lst_of_text
    else:
        new_indexes = np.mean(np.array(list(range(1,len(res)+1))).reshape(-1, topK), axis=1)
        for i,n in enumerate(new_indexes):
            # res.loc[n-topK//2:n+topK//2,'text'] = i+1
            res.loc[n-topK//2:n+topK//2,'text'] = lst_of_text[i]
    res = res[['text','answer','score','start','end']]
    print(res)
    return res

if __name__ == '__main__':
    lst_of_text = ["Shehbaz and Hamza summoned in money laundering case. The Federal Investigation Agency or FIA filed a petition before Judicial Magistrate Kamran Zafar in which it said the statements of 26 accused people would be recorded in the case. The judicial magistrate remarked that this should happen in court.On the other hand, Shehbaz Sharif and Hamza Shahbaz did not appear. Shehbaz had left for Sialkot to attend a workers’ convention.On September 16, Shehbaz Sharif, Hamza Shahbaz and other people accused in the case appeared before the accountability court, while Shehbaz’s wife Nusrat joined the trial through a representative.The court summoned her through a representative to start indictment proceedings at the next hearing.Shehbaz and Hamza have already been indicted.Shehbaz Sharif told the court that Nusrat was out of the country for medical treatment.She will join the proceedings once her treatment is over and he requested the court to allow her to join the proceedings through a representative.NAB has accused Shehbaz Sharif of money laundering and owning more assets than his known sources of income. \nThe evidence collected revealed that Shehbaz and his sons Hamza Shehbaz and Suleman Shahbaz and wife Nusrat Shehbaz had net assets of Rs14.8 million in 1998, according to the accountability bureau.Shehbaz in “connivance with his other family members/benamidars accumulated assets to the tune of Rs7,328 million till 2018”.The amount is “disproportionate to known sources of income and for which neither the accused nor his other family members/benamidars could reasonably account for.” Shehbaz’s lawyer had, however, argued that a reference has been filed and investigation completed, adding that all questionnaires mentioned by NAB have been answered too.",
    "Six bankers record statements against Shehbaz, Hamza in money laundering case. Shehbaz Sharif and his son Hamza Shahbaz.The bankers have recorded their statement under section 164 of the Code of Criminal Procedure in an Rs25 billion money laundering case registered by the Federal Investigation Agency (FIA) against the father and son duo.According to sources, the FIA has presented a list of 26 bankers alleging that the Shehbaz and Hamza opened various bank accounts in the name of security guards, watchmen, clerks working in the Ramzan Sugar Mills.Special Judicial Magistrate Muhammad Kamran Zafar had summoned the witnesses to record their statements today. He had also summoned Shehbaz and Hamza to appear for the cross-examination but were a no show.The magistrate after recording statements of six bankers adjourned the proceedings of the case for September 23 to record statements of the rest of the 20 witnesses. \nAmong the 26 witnesses to record statements against Shahbaz Sharif and Hamza Shahbaz include Maleeha Yousaf of HBL New Muslim Town branch Lahore, Iram Batool Hashmi of MCB Bahawalpur branch, Afzal Ghauri Bhatti of MCB Karim Block, Iqbal Town branch Lahore, Ishfaq Ahmad of MCB Faisalabad branch, Hamid Ali of MCB Mianwali branch, Tayyab Ali of MCB Jhang Road branch, Chiniot, Muhammad Atif Ameer of MCB Sargodha branch, Afzaal Ahmad of MCB Akbar Chowk branch Lahore, Asif Naeem of MCB Jamia Bad branch, Syed Tabbassum Raza Naqvi of MCB Jamia Bad branch, Ahmad Ali of MCB Jhang Road branch, Chiniot, Khurram Shahzad of MCB Chenab Nagar branch, Chiniot, Muhammad Rehman Sarwar of MCB Faisalabad, Fareeha Ijaz of MCB Ajmal House branch, Lahore, Naveed Wahab of UBL, Chiniot branch, Tanvir Hussain of Meezan Bank, Chiniot branch, Asif Iqbal of UBL Chenab Nagar branch, Chiniot, Farrukh Hussain of Meezan Bank of Adda branch, Sargodha, Ghulam Mustafa of Allied Bank main branch Chiniot, Nadeem Ahmad, Kotwali branch Faisalabad, Naseer Ahmad of Allied Bank Faisalabad, Riaz Hussain of Allied Bank, Chiniot branch, Muhammad Zubair of Meezan Bank Lalian branch, Chiniot, Shakeel Ahmad of UBL Sargodha Road branch, Chiniot and Muhammad Hashaam of Meezan Bank, Main branch, Chiniot. \nIt may be relevant to mention here that following investigations into the sugar scam, FIA Lahore had on November 14, 2020, registered an FIR No 39/20 against Shahbaz Sharif, Hamza Shahbaz and Salman Shahbaz for their alleged involvement in Rs 25 billion money laundering which FIA has claimed was committed through various benami accounts.Salman Shahbaz is currently in the UK and did not join the investigations whereas Shahbaz Sharif and Hamza Shahbaz have appeared before the FIA investigators couple of times and are currently on pre-arrest bails"]
    question = "What is the name of person being accused?"

    results = ask_question(lst_of_text, question)
    results['ner_results'] = results['answer'].astype(str).apply(get_multi_ner)
    results['pep_match_results'] = results['ner_results'].apply(check_pep)
    results.to_csv('results.csv')
    print(results)
