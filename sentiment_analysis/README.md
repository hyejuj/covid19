You need to install and train ABSApp first. The code and instructions are here: https://intellabs.github.io/nlp-architect/absa_solution.html

- sentiment_canada.py: inference of sentiment for the canada tweets
- sentiment_us.py: inference of sentiment for the us tweets (line 17, 30-31: these lines allow processing only certain number of tweets. I used these lines so that I could parallerize processing different parts of the data.
- collect_sentiment.py: combine multiple sentiment output after running sentiment_us.py on different parts of data in parallel.
