This repository contains scripts used for Hyeju Jang's COVID-19 paper that analyzes COVID-19 related tweets.

The tweets used for this study are from Chen et al., 2020 (https://publichealth.jmir.org/2020/2/e19273). They have a github repository, and you can download the tweets using their scripts.

Currently, the scripts are not organized for different tasks. To be updated.

* data_statistics: scripts to count frequencies of tweets per certain condition, e.g., location
* Preprocessing: scripts to prepare data by filtering out non-canada, non-us, and non-english.
* topic_modeling: scripts to build a topic model using LDA/NMF, and get topic distributions per location and time(weekly)
* sentiment_analysis: scripts to do inference of sentiments using ABSApp. Training should be done following the instructions on the ABSApp website.

Warning: You need to check input, output, and hardcoded values.

