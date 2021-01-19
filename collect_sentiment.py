import csv
from os import listdir
from os.path import isfile, join

class Aspect:
    term = None
    pos = None
    neg = None
    score = 0
    cnt = 0

nums = [0, 20000, 40000, 60000, 80000, 100000, 120000, 140000, 160000, 
        180000, 200000, 220000, 240000, 260000, 280000]

aspect_dic = {}
for num in nums:
    input_file = 'sentiment_us_%d.csv' % num

    with open(input_file, 'r') as rf:
        reader = csv.DictReader(rf)
        
        for row in reader:
            term = row['term']
            if term in aspect_dic:
                a_class = aspect_dic[term]
                a_class.pos += int(row['pos'])
                a_class.neg += int(row['neg'])
                a_class.score += float(row['score'])
                a_class.cnt += int(row['cnt'])
            else:
                a_class = Aspect()
                a_class.term = term
                a_class.pos = int(row['pos'])
                a_class.neg = int(row['neg'])
                a_class.score = float(row['score'])
                a_class.cnt = int(row['cnt'])
                aspect_dic[term] = a_class
output_file = 'sentiment_us_all.csv'
with open(output_file, 'w') as wf:
    writer = csv.writer(wf)
    headers = ['term', 'pos', 'neg', 'score', 'cnt']
    writer.writerow(headers)
    for term, aspect in aspect_dic.items():
        row = [term, aspect.pos, aspect.neg, aspect.score, aspect.cnt]
        writer.writerow(row)
