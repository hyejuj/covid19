import pendulum
import json
import copy
import gzip
import joblib
from pathlib import Path
import csv
import sys
import pickle
from collections import defaultdict, OrderedDict
import preprocessor as p
from datetime import datetime
from nlp_architect.models.absa.inference.inference import SentimentInference
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer

class Aspect:
    term = None
    pos = 0
    neg = 0
    score = 0
    cnt = 0

ASPECT_LEX = '/home/hyejuj/nlp-architect/cache/absa/train/lexicons/generated_aspect_lex.csv'
OPINION_LEX = '/home/hyejuj/nlp-architect/cache/absa/train/lexicons/generated_opinion_lex_reranked.csv'
keywords = ['asians', 'asian', 'chinese', 'american', 'americans', 'canadian', 'canadians']
#keywords = ['asians', 'asian']

def get_week_of_month(year, month, day):
    target_date = pendulum.datetime(year, month, day)
    week_start = target_date.start_of('week')
    week_end = target_date.end_of('week') 
    bin_name = '%s-%s' % (week_start.strftime('%m/%d'), week_end.strftime('%m/%d'))
    return bin_name

def binning_weekly(input_prefix, input_num):
    bins = defaultdict(list)
    
    p.set_options(p.OPT.URL, p.OPT.RESERVED)
    cnt = 0
    for i in range(input_num):
        input_file = '%s_%d.txt' % (input_prefix, i)
        with open(input_file, 'r') as rf:
            for line in rf:
                # date conversion
                data_json = json.loads(line)
                time_expr = data_json['created_at']
                text = data_json['full_text']
                clean_text = p.clean(text)
                if not set(keywords).intersection(clean_text.lower().split()):
                    continue
                new_time = datetime.strftime(datetime.strptime(time_expr,
                    '%a %b %d %H:%M:%S +0000 %Y'), '%Y-%m-%d')
                year, month, day = [int(x) for x in new_time.split('-')]
                # decide on the name of the bin
                bin_name = get_week_of_month(year, month, day)
                # put text into the bin
                bins[bin_name].append(clean_text)
                cnt += 1
    for i, texts in bins.items():
        print('$', i, len(texts))
    return bins

def get_senti_dist(bins, output_file):
   

    # for each bin, get mean probability
    time_sent = {}
    inference = SentimentInference(ASPECT_LEX, OPINION_LEX)

    with open(output_file, 'w') as f:
        writer = csv.writer(f)
        for i, texts in bins.items():
            aspect_dic = {}
            for tweet in texts:
                sentiment_doc = inference.run(tweet)
                if sentiment_doc == None:
                    continue
                sents = sentiment_doc._sentences
                for sent in sents:
                    if sent == None:
                        continue
                    events = sent._events
                    for event_list in events:
                        if len(event_list) == 0:
                            continue
                        for event in event_list:
                            term = event._text
                            term_type = event._type
                            polarity = event._polarity
                            score = event._score

                            if str(term_type) != 'TermType.ASPECT':
                                continue
                            if term not in keywords:
                                continue
                            if term in aspect_dic:
                                a_class = aspect_dic[term]
                                if str(polarity) == 'Polarity.POS':
                                    a_class.pos += 1
                                elif str(polarity) == 'Polarity.NEG':
                                    a_class.neg += 1
                                else:
                                    print('WRONG SENTIMENT', polarity)
                                a_class.cnt += 1
                                a_class.score += score
                            else:
                                a_class = Aspect()
                                a_class.term = term
                                if str(polarity) == 'Polarity.POS':
                                    a_class.pos = 1
                                elif str(polarity) == 'Polarity.NEG':
                                    a_class.neg = 1
                                else:
                                    print('WRONG SENTIMENT', polarity)
                                a_class.cnt = 1
                                a_class.score = score
                                aspect_dic[term] = a_class
            if len(aspect_dic) > 0:
                for term, a_class in aspect_dic.items():
                    new_row = [i, term, a_class.pos, a_class.neg,
                            a_class.score, a_class.cnt]
                    writer.writerow(new_row)



def run(input_prefix, input_num, output_file):
    tweets_in_bins = binning_weekly(input_prefix, input_num)
    
    get_senti_dist(tweets_in_bins, output_file)

if __name__ == '__main__':
    input_loc = 'canada_us'
    input_prefix = '../COVID-19-TweetIDs/tweets_%s_en' % input_loc# _0.txt'
    input_num = 5 # 5 files
    output_file = 'sentiment_time_%s.csv' % input_loc
    run(input_prefix, input_num, output_file)

