import joblib
import json
import sys
import pickle
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
import pandas as pd
from collections import defaultdict
import operator

def get_texts(input_file):
    texts = []
    with open(input_file) as rf:
        for line in rf:
            texts.append(line.strip())
    print(len(texts))
    return texts

def display_topics(model, feature_names, no_top_words, outputfile):
    with open(outputfile, 'w', encoding='utf-8') as f:
        for topic_idx, topic in enumerate(model.components_):
            #print ("Topic %d:" % (topic_idx))
            f.write(u"Topic %d:\n" % (topic_idx))
            #print (" ".join([feature_names[i]
                #for i in topic.argsort()[:-no_top_words - 1:-1]]))
            f.write(" ".join([feature_names[i]
                for i in topic.argsort()[:-no_top_words - 1:-1]]) + '\n')

def run(texts, num_topics, input_name, method='NMF'):
    df = pd.DataFrame(texts, columns=['Text'])
    
    model = None
    feature_names = None
    vectorizer = None
    if method == 'NMF':  
        # NMF is able to use tf-idf
        # tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf = tfidf_vectorizer.fit_transform(df['Text'])
        print('tfidf feature table size: ', tfidf.shape)
        tfidf_feature_names = tfidf_vectorizer.get_feature_names()
        model = NMF(n_components=num_topics).fit(tfidf)
        feature_names = tfidf_feature_names
        vectorizer = tfidf_vectorizer
    elif method == 'LDA':
        # LDA can only use raw term counts for LDA because it is a probabilistic graphical model
        # tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
        tf_vectorizer = CountVectorizer(stop_words='english')
        tf = tf_vectorizer.fit_transform(df['Text'])
        print('tf feature table size: %s', tf.shape)
        tf_feature_names = tf_vectorizer.get_feature_names()
        model = LatentDirichletAllocation(n_components=num_topics).fit(tf)
        # model = LatentDirichle tAllocation(n_components=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(data)
        feature_names = tf_feature_names
        vectorizer = tf_vectorizer
    
    output_file = '%s_%s_%d.txt'%(input_name, method, num_topics)
    display_topics(model, feature_names, 50, output_file)
    joblib.dump([model, feature_names, vectorizer.vocabulary_], 
            '%s_%s_%d.model'%(input_name, method, num_topics),
            compress=1)
    return model

def display_documents(texts, model, no_sents, voca, method, output_file):
    df = pd.DataFrame(texts, columns=['Text'])

    doc_topic_dist = None
    if method == 'LDA':
        tf_vectorizer = CountVectorizer(stop_words='english', 
                vocabulary=voca)
        tf = tf_vectorizer.fit_transform(df['Text'])
        doc_topic_dist = model.transform(tf)
    elif method == 'NMF':
        tfidf_transformer = TfidfTransformer()
        loaded_vec = CountVectorizer(stop_words='english',
                vocabulary=voca)
        tfidf = tfidf_transformer.fit_transform(loaded_vec.fit_transform(df['Text']))
        doc_topic_dist = model.transform(tfidf)
    else:
        print('wrong method:', method)
    topic_docs = defaultdict(dict)
    for n in range(doc_topic_dist.shape[0]):
        topic_most_pr = doc_topic_dist[n].argmax()
        #print('##', doc_topic_dist[n])
        #print('doc: {} topic: {}\n'.format(n, topic_most_pr))
        #print(doc_topic_dist[n][topic_most_pr])
        topic_docs[topic_most_pr][n] = \
                doc_topic_dist[n][topic_most_pr]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for topic_idx in range(doc_topic_dist.shape[1]):
            #print ("Topic %d:" % (topic_idx))
            f.write(u"Topic %d:\n" % (topic_idx))
            # sort dictionary items
            sorted_docs = sorted(topic_docs[topic_idx].items(),
                    key=operator.itemgetter(1), reverse=True)
            for i in range(no_sents):
                #print('##', sorted_docs[i])
                #print(df['Text'].iloc[sorted_docs[i][0]])
                f.write('#%d: %s\n\n'%(i+1, df['Text'].iloc[sorted_docs[i][0]]))

def load(texts, num_topics, input_name, method='NMF'):
    model_file = '%s_%s_%d.model'%(input_name, method, num_topics)
    print('%s is loaded' % model_file)
    model, feature_names, voca = joblib.load(model_file)
    output_file = '%s_%s_%d_sents.txt'%(input_name, method, num_topics)
    display_topics(model, feature_names, 50, output_file)
    display_documents(texts, model, 3, voca, method, output_file)

if __name__ == '__main__':
    input_file = 'canada_us.txt'
    num_topics = int(sys.argv[1]) # 20
    method = sys.argv[2] #'LDA', 'NMF'
    texts = get_texts(input_file)
    input_name = input_file[:input_file.index('.txt')]
    run(texts, num_topics, input_name, method=method)
    
    load(texts, num_topics, input_name, method=method)
