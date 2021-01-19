import sys

def get_stopwords():
    stopwords = None
    gist_file = open('gist_stopwords.txt', 'r')
    try:
        content = gist_file.read()
        stopwords = content.split(',')
    finally:
        gist_file.close()
    return stopwords

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
            print ("Topic %d:" % (topic_idx))
            f.write(u"Topic %d:\n" % (topic_idx))
            print (" ".join([feature_names[i]
                for i in topic.argsort()[:-no_top_words - 1:-1]]))
            f.write(" ".join([feature_names[i]
                for i in topic.argsort()[:-no_top_words - 1:-1]]) + '\n')


def run(texts, keywords, output_file):
    vocab = {}
    stopwords = get_stopwords()
    writer = open(output_file, 'w')
    for text in texts:
        tokens = text.lower().split()
        if set(keywords).intersection(tokens):
            writer.write(text + '\n')
            '''
            for tok in tokens:
                if (tok in stopwords) or (tok in keywords):
                    continue
                if tok in vocab:
                    vocab[tok] += 1
                else:
                    vocab[tok] = 1

    ''''''
    sorted_vocab = sorted(vocab.items(), key=lambda item: item[1], reverse=True)
    for w in sorted_vocab[:100]:
        print(w)
    '''
    writer.close()
        


if __name__ == '__main__':
    input_file = 'canada_us.txt'
    output_file = 'canada_us_asians.txt'
    keywords = ['asian', 'asians', 'immigrants', 'immigrant', 'chinese']
    texts = get_texts(input_file)
    run(texts, keywords, output_file)
    
