from nlp_architect.models.absa.inference.inference import SentimentInference
import csv
import sys

class Aspect:
    term = None
    pos = 0
    neg = 0
    score = 0
    cnt = 0

ASPECT_LEX = '/home/hyejuj/nlp-architect/cache/absa/train/lexicons/generated_aspect_lex.csv'
OPINION_LEX = '/home/hyejuj/nlp-architect/cache/absa/train/lexicons/generated_opinion_lex_reranked.csv'

input_file = '../COVID-19-TweetIDs/us.txt'
#input_file = 'temp.txt'
num = int(sys.argv[1])
output_file = 'sentiment_us_%d.csv' % num
with open(input_file, 'r') as rf, open(output_file, 'w') as wf:
    inference = SentimentInference(ASPECT_LEX, OPINION_LEX)
    writer = csv.writer(wf)
    headers = ['term', 'pos', 'neg', 'score', 'cnt']
    writer.writerow(headers)
    aspect_dic = {}
    cnt = 0
    for tweet in rf:
        if cnt < num:
            cnt += 1
            continue
        if cnt > num + 20000:
            break
        print(cnt, tweet)
        sentiment_doc = inference.run(tweet)
        if sentiment_doc == None:
            cnt += 1
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
                            print('WRONG SENTIMENT2', polarity)
                        a_class.cnt = 1
                        a_class.score = score
                        aspect_dic[term] = a_class
        cnt += 1
    for term, a_class in aspect_dic.items():
        new_row = [term, a_class.pos, a_class.neg, 
                a_class.score, a_class.cnt]
        writer.writerow(new_row)
