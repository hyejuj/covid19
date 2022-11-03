import json
import csv
import sys
import pickle
import spacy
from spacy_langdetect import LanguageDetector
nlp = spacy.load('en')
nlp.add_pipe(LanguageDetector(), name='language_detector', 
        last=True)

import preprocessor as p

def get_eng_texts(input_file, output_file):
    with open(input_file, 'r') as rf, open(output_file, 'w') as wf:
        p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.RESERVED)
        for line in rf:
            data_json = json.loads(line)
            text = data_json['full_text']
            clean_text = p.clean(text)
            doc = nlp(clean_text)
            lang = doc._.language['language']
            if lang == 'en':
                wf.write(line.decode('utf8'))
    
if __name__ == '__main__':
    month_index = int(sys.argv[1])
    input_file = 'tweets_us_%d.txt' % month_index
    output_file = 'tweets_us_en_%d.txt' % month_index
    
    get_eng_texts(input_file, output_file)



