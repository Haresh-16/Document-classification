"""
SpaCy for keyword extraction from input complaint text!
"""

import spacy

from collections import Counter
from string import punctuation
import en_core_web_md

nlp = en_core_web_md.load()

def getKeywords(text):
    result=[]
    
    POSTAGS=['PROPN','ADJ','NOUN']
    
    doc=nlp(text.lower())
    
    for token in doc:
        
        if(token.text in nlp.Defaults.stop_words or token.text in punctuation):
            
            continue
        
        if(token.pos_ in POSTAGS):
            
            result.append(token.text)
    
    return result

text = input()

print(getKeywords(text))
            
            
            
            
    
    