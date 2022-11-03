import json
import csv
import sys
import pickle

import preprocessor as p

def get_texts(input_file, output_file):
    with open(input_file, 'r') as rf, open(output_file, 'w') as wf:
        p.set_options(p.OPT.URL, p.OPT.RESERVED)
        for line in rf:
            data_json = json.loads(line)
            text = data_json['full_text']
            clean_text = p.clean(text)
            wf.write(clean_text + '\n')
    
if __name__ == '__main__':
    month_index = int(sys.argv[1])
    input_file = 'tweets_us_en_%d.txt' % month_index
    output_file = 'us_clean_%d.txt' % month_index
    
    texts = get_texts(input_file, output_file)



