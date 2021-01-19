import pendulum
import calendar
import json
import gzip
import joblib
from pathlib import Path
import csv
import numpy as np
import sys
import pickle
from collections import defaultdict, OrderedDict
import preprocessor as p
from datetime import datetime

from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer

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

def get_topic_dist(bins, method, model_file, output_file):
    # load LDA/NMF model
    model, feature_names, voca = joblib.load(model_file)
   

    # for each bin, get mean probability
    time_topic = {}

    for i, texts in bins.items():
        doc_topic_dist = None
        if method == 'LDA':
            tf_vectorizer = CountVectorizer(stop_words='english', 
                    vocabulary=voca)
            tf = tf_vectorizer.fit_transform(texts)
            doc_topic_dist = model.transform(tf)
        elif method == 'NMF':
            tfidf_transformer = TfidfTransformer()
            loaded_vec = CountVectorizer(stop_words='english',
                    vocabulary=voca)
            tfidf = tfidf_transformer.fit_transform(loaded_vec.fit_transform(texts))
            doc_topic_dist = model.transform(tfidf)
        else:
            print('wrong method:', method)
    
        # compute average
        avg = doc_topic_dist.mean(axis=0)
        time_topic[i] = avg.tolist()

    sorted_dic = {k: time_topic[k] for k in sorted(time_topic)}
    with open(output_file, 'w') as f:
        writer = csv.writer(f)
        for k, v in sorted_dic.items():
            new_row = [k] + v
            writer.writerow(new_row)


def run(input_prefix, input_num, method, model_file, output_file):
    tweets_in_bins = binning_weekly(input_prefix, input_num)
    
    get_topic_dist(tweets_in_bins, method, model_file, output_file)

if __name__ == '__main__':
    input_loc = 'canada'
    input_prefix = 'tweets_%s_en' % input_loc# _0.txt'
    method = 'LDA'
    model_loc = 'canada_us'
    model_file = '%s_%s_20.model' % (model_loc, method) # model could be different from input e.g., doesn't need to be canada
    input_num = 5 # 5 files
    output_file = 'time_%s_%s_%s.csv' % (input_loc, model_loc, method) 
    run(input_prefix, input_num, method, model_file, output_file)

