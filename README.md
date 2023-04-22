# USTORY


## Libraries
- pandas 1.2.4
- sklearn 0.24.2
- numpy 1.19.5
- scipy 1.4.1
- sentence-transformers 2.0.0
- torch 1.8.2 (for sentence-transformer)
- spherical_kmeans ([source](https://github.com/rfayat/spherecluster/blob/scikit_update/spherecluster/spherical_kmeans.py))
- b3 ([source](https://github.com/m-wiesner/BCUBED/blob/master/B3score/b3.py))

## Data sets
### Raw data sets (before preprocessed)
- Data sets link: [link](https://www.dropbox.com/sh/1dojygynitafqid/AAA0xko654RyJbmyLk3kgGH6a?dl=0)
  - Newsfeed14 [source](https://github.com/Priberam/news-clustering/blob/master/download_data.sh)
  - WCEP18/19 [source](https://github.com/complementizer/wcep-mds-dataset)
  - USNews (case study) - included in the link

### Preprocessing
1. Download the raw data sets in the above link, or you can prepare your own data set ('.csv','.json',..) where the row format is ['title', 'date', 'text', 'id', 'story' (if available)]
3. Run Dataset_preprocessing.ipynb to preprocess the data set

## USTORY usage
### Parameters
- file_path: the path to a preprocessed data set file
- window_size: the size of window in desired time units (e.g., days) - default = 7
- slide_size: the size of slide in desired time units (e.g., days) - default = 1
- num_windows: the total number of windows to evaluate - default = 365
- min_articles: the minimum number of articles to form a story (e.g., 8 for Newsfeed14 and 18 for WCEP18/19 and USNews by default)
- N: the number of thematic keywords - default = 10
- T: the temperature for scaling the confidence score - default: 2
- keyword_score: the type of keyword score function in ["tfidf", "bm25"] - default = "tfidf"
- verbose: print the intermediate process in ["True", "False"] - default = "False"
- story_label: the existence of label for data set (to evaluate accuracy) in ["True", "False"] - default = "True"

### Run an example simulation

#### Output items (in order)
(all_window, cluster_keywords_df, final_num_cluster, avg_win_proc_time, nmi, ami, ri, ari, precision, recall, fscore)
- all_window: include all article information and cluster assignment/confidence information
- cluster_keywords_df: the lists of thematic keywords of clusters in every window

```
from USTORY import *
output = simulate(file_path, window_size, slide_size, begin_date, num_windows, min_articles, N, T, keyword_score, verbose, story_label)
print(fscore: output[-1])
```

