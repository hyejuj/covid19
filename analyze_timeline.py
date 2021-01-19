import json
import gzip
import joblib
from pathlib import Path
import csv
import sys
import pickle
from collections import defaultdict, OrderedDict
import preprocessor as p
from datetime import datetime

from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer


def binning(input_prefix, input_num, num_bin_per_month=1):
    splitters = []
    days_bin = None
    if num_bin_per_month > 1:
        days_bin = int(30 / num_bin_per_month)
        for i in range(num_bin_per_month): 
            splitters.append(i*days_bin)
    
    bin_num = input_num * num_bin_per_month
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
                new_time = datetime.strftime(datetime.strptime(time_expr,'%a %b %d %H:%M:%S +0000 %Y'), '%m-%d')
                month, day = [int(x) for x in new_time.split('-')]
                # decide on the name of the bin
                bin_name = None
                for splitter in splitters:
                    if day <= splitter:
                        bin_name = '%d:%d-%d' % (month, splitter-days_bin, splitter)
                        break
                if bin_name == None:
                    bin_name = '%d:%d-%d' % (month, 30-days_bin, 30)
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


def run(input_prefix, input_num, method, model_file, output_file, num_bin_per_month=1):
    tweets_in_bins = binning(input_prefix, input_num, num_bin_per_month)
    
    get_topic_dist(tweets_in_bins, method, model_file, output_file)

if __name__ == '__main__':
    input_loc = 'us'
    input_prefix = 'tweets_%s_en' % input_loc# _0.txt'
    method = 'NMF'
    model_loc = 'canada_us'
    model_file = '%s_%s_20.model' % (model_loc, method) # model could be different from input e.g., doesn't need to be canada
    input_num = 5 # 5 files
    output_file = 'time_%s_%s_%s.csv' % (input_loc, model_loc, method) 
    num_bin = 3 # how many bins per month 
    run(input_prefix, input_num, method, model_file, output_file, num_bin)

