import json
import gzip
from pathlib import Path
import csv
import sys
import pickle
import spacy
from spacy_langdetect import LanguageDetector
nlp = spacy.load('en')
nlp.add_pipe(LanguageDetector(), name='language_detector',
        last=True)

import preprocessor as p

def parse(input_file):
    data = []
    with open(input_file) as rf:
        for line in rf:
            data_json = json.loads(line)
            data.append(data_json)
    return data



def get_hashtag_freq(data):
    hashtags_cnts = {}
    for datum in data:
        hashtags = datum['entities']['hashtags']
        for tag_st in hashtags:
            tag = tag_st['text'].lower()
            if tag in hashtags_cnts:
                hashtags_cnts[tag] += 1
            else:
                hashtags_cnts[tag] = 1

        #[{'text': 'Coronavirus', 'indices': [21, 33]}]
    sorted_cnts = {k: v for k, v in sorted(hashtags_cnts.items(), 
            key=lambda item: item[1], reverse=True)}
    print(sorted_cnts)
    with open('hashtags_cnts.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['hashtag', 'cnt'])
        for k, v in sorted_cnts.items():
            writer.writerow([k,v])

def get_loc_freq(data):
    loc_cnts = {}
    for datum in data:
        loc = datum['bio_location'] #['country']
        if loc == None: 
            continue
        print(loc)
        #loc = '%s, %s' % (loc['full_name'], loc['country']) 
        #loc = loc['country']
        if loc in loc_cnts:
            loc_cnts[loc] += 1
        else:
            loc_cnts[loc] = 1

    sorted_cnts = {k: v for k, v in sorted(loc_cnts.items(), 
            key=lambda item: item[1], reverse=True)}
    print(sorted_cnts)
    with open('place_cnts.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['place', 'cnt'])
        for k, v in sorted_cnts.items():
            writer.writerow([k,v])

def get_lang_freq(data):
    lang_cnts = {}
    for datum in data:
        lang = datum['lang'] #['country']
        if lang == None: 
            continue
        if lang in lang_cnts:
            lang_cnts[lang] += 1
        else:
            lang_cnts[lang] = 1

    sorted_cnts = {k: v for k, v in sorted(lang_cnts.items(), 
            key=lambda item: item[1], reverse=True)}
    print(sorted_cnts)
    with open('lang_cnts.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['lang', 'cnt'])
        for k, v in sorted_cnts.items():
            writer.writerow([k,v])

def get_stopwords(filename):
    with open(filename, 'r') as f:
        stopwords = []
        for line in f:
            w = line.strip()
            stopwords.append(w)
        return stopwords

def get_word_freq(data):
    stopwords = get_stopwords('stopwords.txt')
    texts = get_texts(input_file)
    word_cnts = {}
    for text in texts:
        #text = datum['text']
        tokens = text.split(' ')
        for tok in tokens:
            tok = tok.strip()
            if tok in stopwords:
                continue
            if tok in word_cnts:
                word_cnts[tok] += 1
            else:
                word_cnts[tok] = 1
    sorted_cnts = {k: v for k, v in sorted(word_cnts.items(), 
            key=lambda item: item[1], reverse=True)}
    print(sorted_cnts)
    with open('token_cnts_nostopwords.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['token', 'cnt'])
        for k, v in sorted_cnts.items():
            writer.writerow([k,v])

def select_en(path, wf, cnt_total):
    print(path.name)
    with gzip.open(path, 'rb') as rf:
        for line in rf:
            cnt_total += 1
            data_json = json.loads(line)
            text = data_json['full_text']
            clean_text = p.clean(text)
            doc = nlp(clean_text)
            lang = doc._.language['language']
            if lang == 'en':
                wf.write('%s\n' % clean_text)
    return cnt_total

def get_en(data_dirs, output_file):
    cnt_total = 0
    with open(output_file, 'w') as wf:
        for data_dir in data_dirs:
            for path in Path(data_dir).iterdir():
                if path.name.endswith('.jsonl.gz'):
                    cnt_total = select_en(path, wf, cnt_total)
    print(cnt_total)

if __name__ == '__main__':
    data_dirs = ['2020-01', '2020-02', '2020-03', 
            '2020-04', '2020-05'] 
    month_index = int(sys.argv[1])
    month = [data_dirs[month_index]]
    output_file = 'tweets_en_%d.txt' % month_index
    get_en(month, output_file)
