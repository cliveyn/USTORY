import pandas as pd
import numpy as np
import time
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from scipy.spatial import distance
from scipy.sparse import vstack
import torch
from sentence_transformers import SentenceTransformer

from spherical_kmeans import SphericalKMeans
import b3

from warnings import simplefilter # import warnings filter and ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


def simulate(file_path, window_size, slide_size, num_windows,
             min_articles, N, T, keyword_score, verbose, story_label,
             time_aware = True, theme_aware = True):

    article_df, all_vocab = read_dataset(file_path, story_label, verbose)
    begin_date = article_df.date[0].strftime("%Y-%m-%d")
    
    all_window, window, cluster_keywords_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    window_indices, article_df_slides = [], []
    eval_metrics, win_proc_times = [], []       
    
    article_df['cluster'] = -1 
    article_df['sim'] = 0.0
    cluster_centers = []

    for i in range(num_windows):
        init_start_time = time.time()
        if verbose: print("<Window "+str(i)+">")

        # [1] Get new slide and remove old slide from window
        from_date = pd.to_datetime(begin_date) + pd.DateOffset(days=i*slide_size)
        to_date = pd.to_datetime(begin_date) + pd.DateOffset(days=(i+1)*slide_size)
        slide = article_df[(article_df['date'] >= from_date) & (article_df['date'] < to_date)].copy()   
        
        if len(window_indices) >= window_size/slide_size:
            all_window = pd.concat([all_window, window.loc[window_indices[0]]])
            window.drop(index = window_indices[0], inplace=True)
            window_indices.pop(0)
            article_df_slides.pop(0)
        
        ## Update window indices and article df slides
        if len(slide) < 1:
            article_df_slide = np.zeros(len(all_vocab)).reshape(1,-1)
        else:
            article_TFs = vstack(slide['article_TF'])
            article_df_slide = np.bincount(article_TFs.indices, minlength=article_TFs.shape[1]).reshape(1,-1)

        window_indices.append(slide.index)
        article_df_slides.append(article_df_slide)    
 
        ## Update default article embedding
        slide, article_embedding_time = get_article_embedding(slide, window, article_df_slides, 
                                                              time_aware, theme_aware, 
                                                              keyword_score, N)

        ## Add new slide to window
        window = pd.concat([window, slide])
        
        # [2] Assign to clusters
        ## Initialize cluster centers
        if len(cluster_centers) == 0: 
            num_new_clusters = int(len(window)/min_articles)
            if num_new_clusters < 1: continue
            clustering = SphericalKMeans(n_clusters=num_new_clusters).fit(list(window['embedding'].values)) 
            cluster_centers = clustering.cluster_centers_.tolist()
            if verbose: print(str(len(cluster_centers))+" clusters are initialized")
            cluster_emb_sum_dics = [{} for j in range(len(cluster_centers))]
            cluster_tf_sum_dics = [{} for j in range(len(cluster_centers))]
            cluster_topN_probs = {}

            ## Assign to clusters ##
            initial = True
            window, cluster_emb_sum_dics, cluster_tf_sum_dics, assign_time = assign_to_clusters(initial, verbose,
                                                                  window, window_size, to_date, cluster_centers, 
                                                                  cluster_emb_sum_dics, cluster_tf_sum_dics, cluster_topN_probs, T)
        ## After initialization
        elif len(set(window[window['cluster']>=0]['cluster'])) > 0 and len(window[window['cluster']==-1]) > 0:
            initial = False
            window, cluster_emb_sum_dics, cluster_tf_sum_dics, assign_time = assign_to_clusters(initial, verbose,
                                            window, window_size, to_date, cluster_centers, 
                                            cluster_emb_sum_dics, cluster_tf_sum_dics, cluster_topN_probs, T,
                                            time_aware, theme_aware, 
                                            cluster_topN_indices, cluster_topN_scores)

        ## [3] Cluster outliers 
        window, cluster_centers, cluster_emb_sum_dics, cluster_tf_sum_dics, cluster_time = cluster_outliers(
                                                                        window, cluster_centers, 
                                                                        cluster_emb_sum_dics, cluster_tf_sum_dics, 
                                                                        min_articles, verbose) 

        ## [4] Derive cluster top keywords
        if len(set(window[window['cluster']>=0]['cluster'])) > 0:
            cluster_topN_indices, cluster_topN_scores, cluster_topN_probs, cluster_keyword_time = get_cluster_theme(
                                                                            window, window_size, to_date, 
                                                                            time_aware, 
                                                                            cluster_tf_sum_dics, keyword_score, N)

        # Log statistics

        if len(window) > 0:
            if i >= window_size and story_label:
                eval_metrics.append(eval_metric(window.story.values, window.cluster.values))
            win_proc_times.append(time.time() - init_start_time)
            cluster_keywords_df = update_cluster_keywords_articles(i, window, all_vocab, cluster_keywords_df, cluster_topN_indices)

    all_window = pd.concat([all_window,window])
    
    #For landmark evaluation
    if story_label:
        nmi, ami, ri, ari, precision, recall, fscore = [np.round(k,3) for k in np.mean(eval_metrics,axis=0)]
    else:
        nmi, ami, ri, ari, precision, recall, fscore = 0, 0, 0, 0, 0, 0, 0
    final_num_cluster = len(cluster_centers)
    avg_win_proc_time = np.round(np.mean(win_proc_times),1)
    
    return (all_window, cluster_keywords_df,
            final_num_cluster, avg_win_proc_time,
            nmi, ami, ri, ari, precision, recall, fscore)



def read_dataset(file_name, story_label, verbose):
    article_df = pd.read_json(file_name)
    
    article_df['sentence_embds'] = [np.array(x) for x in article_df['sentence_embds']]

    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2), tokenizer=lambda x: x, lowercase=False, norm=None)
    tfidf_vectorizer.fit_transform([sum(k, []) for k in article_df['sentence_tokens']])
    all_vocab = tfidf_vectorizer.get_feature_names()

    count_vectorizer = CountVectorizer(tokenizer=lambda x: x, ngram_range = (1,2), vocabulary = list(all_vocab), lowercase=False)
    article_df['sentence_TFs'] = [count_vectorizer.transform(y) for y in article_df['sentence_tokens'].values]
    article_df['article_TF'] = [sum(a) for a in article_df['sentence_TFs'].values]
    
    if verbose:
        print(f'{file_name} loaded')
        print(f'articles:{len(article_df)}')
        if story_label:
            print(f'#stories:{len(article_df.story.unique())}')
    
    return article_df, all_vocab


def eval_metric(label, cluster):
    nmi = np.round(metrics.normalized_mutual_info_score(label, cluster),3)
    ami = np.round(metrics.adjusted_mutual_info_score(label, cluster),3)
    ri = np.round(metrics.rand_score(label, cluster),3)
    ari = np.round(metrics.adjusted_rand_score(label, cluster),3)
    fscore, precision, recall = [np.round(k,3) for k in b3.calc_b3(label,cluster)]
    
    return [nmi, ami, ri, ari, precision, recall, fscore]


def get_article_embedding(slide, window, article_df_slides, time_aware, theme_aware, keyword_score, N):
    start_time = time.time()
    if len(slide) < 1:
        return slide, time.time() - start_time      
     
    if theme_aware:
        num_articles = len(window) if len(window) > 0 else len(slide)
        
        if time_aware: #exponential decaying document frequency
            article_df_window = 0 
            for t in range(len(article_df_slides)):
                article_df_window += np.exp(-(len(article_df_slides)-t-1)/len(article_df_slides))*article_df_slides[t]
        else: #document frequency
            article_df_window = np.sum(article_df_slides,axis=0) 
        
        article_idf_window = np.log((num_articles+1)/(article_df_window+1))+1 #inverse document frequency - scikit-learn formual = log((N+1)/(df+1))+1
        article_tf_window = vstack(slide['article_TF'].values) #term frequency
              
        if keyword_score == 'tfidf':
            article_keyword_score_all = article_tf_window.multiply(article_idf_window).tocsr()
        elif keyword_score == 'bm25':
            k1 = 1.2
            b = 0.75
            d = 1.0
            
            avgDL =  np.sum(vstack(window['article_TF'].values))/num_articles if len(window) > 0 else np.sum(vstack(slide['article_TF'].values))/num_articles #average document length
            article_ntf_window = article_tf_window.multiply(1/np.array(1-b+b*np.sum(article_tf_window,axis=1)/avgDL)) # normalized term frequency - pivoted length normalization - eq3 in Yuanhua 2011
            article_ntf_window.data = article_ntf_window.data # shifting - eq4 in Yuanhua 2011
            article_ntf_window.data = ((k1 + 1) * article_ntf_window.data)  / (k1 + article_ntf_window.data)  + d # tf normalization - eq4 in Yuanhua 2011
            article_keyword_score_all = article_ntf_window.multiply(article_idf_window).tocsr()
        
    weighted_embs = []
    num_processed_articles = 0
    for (idx,article) in slide.iterrows():
        if theme_aware:
            article_topN_indices = article_keyword_score_all[num_processed_articles].indices[article_keyword_score_all[num_processed_articles].data.argsort()[:-(N+1):-1]]
            article_topN_scores = article_keyword_score_all[num_processed_articles][:,article_topN_indices]
            sentence_raw_weights = np.array(np.sum(article.sentence_TFs[:,article_topN_indices].multiply(article_topN_scores), axis=1)).ravel() + 1e-5
            sentence_weights = sentence_raw_weights / np.sum(sentence_raw_weights, axis=0)
            num_processed_articles += 1
        else:
            num_sentences = len(article['sentences'])
            sentence_weights = [1/num_sentences]* num_sentences 
        weighted_embs.append(np.matmul(sentence_weights,article.sentence_embds))
    
    slide['embedding'] = weighted_embs
    
    return slide, time.time() - start_time


def get_cluster_theme(window, window_size, to_date, time_aware, cluster_tf_sum_dics, keyword_score, N):
    start_time = time.time()
    cluster_ids = list(set(window[window['cluster']>=0]['cluster']))
    
    cluster_tf_dic = {}
    for cluster_id in cluster_ids:
        if time_aware: #exponential decaying cluster tf
            cluster_tf_dic[cluster_id] = 0
            decaying_factor = window_size
            for date in sorted(cluster_tf_sum_dics[cluster_id].keys())[::-1]: #sorted by newest -> oldest
                time_delta = (to_date - date).days-1
                if time_delta >= window_size: break # only consider window context
                cluster_tf_dic[cluster_id] += np.exp(-time_delta/decaying_factor)*cluster_tf_sum_dics[cluster_id][date] 
        else: #normal cluster tf
            cluster_tf_dic[cluster_id] = np.sum(window[window['cluster']==cluster_id].article_TF)  
            
    cluster_tf = vstack(cluster_tf_dic.values())
    cluster_df = np.bincount(cluster_tf.indices, minlength=cluster_tf.shape[1]).reshape(1,-1)
    cluster_idf = np.log((len(cluster_ids)+1)/(cluster_df+1))+1 #scikit-learn formual = log((N+1)/(df+1))+1
    
    if keyword_score == 'tfidf':
        cluster_keyword_score_all = cluster_tf.multiply(cluster_idf).tocsr()
    elif keyword_score == 'bm25':
        k1 = 1.2
        b = 0.75
        d = 1.0
        avgDL = np.sum(cluster_tf)/len(cluster_ids)        
        cluster_ntf = cluster_tf.multiply(1/np.array(1-b+b*np.sum(cluster_tf,axis=1)/avgDL))
        cluster_ntf.data = ((k1 + 1) * cluster_ntf.data)  / (k1 + cluster_ntf.data) + d# tf normalization - eq4 in Yuanhua 2011
        cluster_keyword_score_all = cluster_ntf.multiply(cluster_idf).tocsr()
    
    cluster_topN_indices = {}
    cluster_topN_scores = {}
    cluster_topN_probs = {}
    for i in range(len(cluster_ids)):
        cluster_id = cluster_ids[i]
        cluster_topN_indices[cluster_id] = cluster_keyword_score_all[i].indices[cluster_keyword_score_all[i].data.argsort()[:-(N+1):-1]]
        cluster_topN_scores[cluster_id] = cluster_keyword_score_all[i][:,cluster_topN_indices[cluster_id]]
        cluster_topN_tfs = cluster_tf_dic[cluster_id][:,cluster_topN_indices[cluster_id]]
        cluster_topN_probs[cluster_id] = [np.round(x,5) for x in (cluster_topN_tfs/np.sum(cluster_topN_tfs)).toarray()[0]] #rounding to avoid scipy js error (approximately same probs return na)
        
    return cluster_topN_indices, cluster_topN_scores, cluster_topN_probs, time.time() - start_time


def assign_to_clusters(initial, verbose, window, window_size, to_date, cluster_centers, 
                       cluster_emb_sum_dics, cluster_tf_sum_dics, cluster_topN_probs,
                       T, time_aware = False, theme_aware = False, 
                       cluster_topN_indices = None, cluster_topN_scores = None):
    
    start_time = time.time()

    if initial:
        considered_center_indices = list(range(len(cluster_centers)))
    else:
        considered_center_indices = list(set(window[window['cluster']>=0]['cluster']))

    if verbose: print("Assign to "+str(len(considered_center_indices))+" clusters")
    out_thred = (1-1/(len(considered_center_indices)+1))**T #+1 to handle a single cluster

    if theme_aware:
        sentence_tfs_all = vstack(window[window.cluster==-1]['sentence_TFs'].values)
        article_tfs_all = vstack(window[window.cluster==-1]['article_TF'].values)
        sentence_raw_weights_all = {}
        article_topN_tfs_all = {}
        for cluster_id in considered_center_indices:
            sentence_raw_weights_all[cluster_id] = np.array(np.sum(sentence_tfs_all[:,cluster_topN_indices[cluster_id]].multiply(cluster_topN_scores[cluster_id]), axis=1)).ravel()                       
            article_topN_tfs_all[cluster_id] = article_tfs_all[:,cluster_topN_indices[cluster_id]].toarray()
            
    if time_aware:
        time_weighted_center_dic = {}
        for uniq_date in window[window.cluster==-1].date.unique():
            for cluster_id in considered_center_indices:
                time_weighted_sum = 0
                time_weighted_num = []
                
                decaying_factor = window_size
                #decaying_factor = len(cluster_emb_sum_dics[cluster_id])
                for date in sorted(cluster_emb_sum_dics[cluster_id].keys())[::-1]: #sorted by newest -> oldest time
                    if (to_date - date).days-1 >= window_size: break #consider only the window context
                    day_delta = np.abs((uniq_date - date).days)
                    time_weighted_num.append(np.exp(-day_delta/decaying_factor)*cluster_emb_sum_dics[cluster_id][date][1]) # time+amount weighted's average
                    time_weighted_sum += np.exp(-day_delta/decaying_factor) * cluster_emb_sum_dics[cluster_id][date][0] 
                time_weighted_center = time_weighted_sum/sum(time_weighted_num)
                time_weighted_center_dic[(pd.Timestamp(uniq_date), cluster_id)] = time_weighted_center

            
    num_processed_articles = 0
    num_processed_sentences = 0
    for (idx,article) in window[window.cluster==-1].iterrows():
        w_emb = article.embedding # default article embedding
        
        ## Evaluate the similarity to clusters
        if theme_aware:
            similarities = []
            total_weighted_embeddings = []

            for cluster_id in considered_center_indices:                
                sentence_raw_weights = np.array(sentence_raw_weights_all[cluster_id][num_processed_sentences:num_processed_sentences + len(article.sentences)]).ravel()
                if sum(sentence_raw_weights) > 0:
                    sentence_weights = sentence_raw_weights / np.sum(sentence_raw_weights)
                    c_emb = np.matmul(sentence_weights,article.sentence_embds)
                    
                    total_weighted_emb = c_emb
                else: #if any of sentence is weighted, then just use default embedding
                    total_weighted_emb = w_emb 

                total_weighted_embeddings.append(total_weighted_emb)

                if time_aware:
                    time_weighted_center = time_weighted_center_dic[(article['date'], cluster_id)]
                    cos_sim = np.dot(total_weighted_emb, time_weighted_center)/(np.linalg.norm(total_weighted_emb)*np.linalg.norm(time_weighted_center))
                else:
                    cos_sim = np.dot(total_weighted_emb, cluster_centers[int(cluster_id)])/(np.linalg.norm(total_weighted_emb)*np.linalg.norm(cluster_centers[int(cluster_id)]))
                
                if sum(sentence_raw_weights) > 0:
                    article_topN_tfs = article_topN_tfs_all[cluster_id][num_processed_articles]
                    p_cluster  = cluster_topN_probs[cluster_id]
                    p_article = (article_topN_tfs/np.sum(article_topN_tfs))
                    js_sim = 1 - distance.jensenshannon(p_cluster,p_article)
                else:
                    js_sim = 0
                
                if cos_sim < 0: cos_sim = 0
                similarities.append(cos_sim*js_sim)
            num_processed_sentences += len(article.sentences)
            num_processed_articles += 1
        else:
            if time_aware:
                similarities = []
                for cluster_id in considered_center_indices:
                    time_weighted_center = time_weighted_center_dic[(article['date'], cluster_id)]
                    cos_sim = np.dot(article.embedding, time_weighted_center)/(np.linalg.norm(article.embedding)*np.linalg.norm(time_weighted_center))
                    similarities.append(cos_sim)
            else:
                considered_centers = [cluster_centers[int(k)] for k in considered_center_indices]
                similarities = cosine_similarity([article.embedding], considered_centers)
        
        probs = np.exp(T*np.array(similarities)).ravel()
        probs = probs/np.sum(probs)
       
        ## Assign to the most appropriate cluster
        if not initial and len(probs) < 2:
            conf = np.max(similarities) #if a single cluster
        else:
            conf = np.max(probs)
        if 1-conf > out_thred:
            window.at[idx,'cluster'] = -1
            window.at[idx,'sim'] = 0
        else:
            cluster_id = considered_center_indices[np.argmax(probs)]
            window.at[idx,'cluster'] = cluster_id
            window.at[idx,'sim'] = np.max(probs)
            
            if theme_aware: #update embedding
                window.at[idx,'embedding'] = total_weighted_embeddings[np.argmax(probs)]

            if article['date'] not in cluster_emb_sum_dics[cluster_id]:
                cluster_emb_sum_dics[cluster_id][article['date']] = [0,0]
                cluster_tf_sum_dics[cluster_id][article['date']] = 0
            cluster_emb_sum_dics[cluster_id][article['date']][0] += article['embedding'] #embedding sum
            cluster_emb_sum_dics[cluster_id][article['date']][1] += 1 #article count
            cluster_tf_sum_dics[cluster_id][article['date']] += article['article_TF'] #article tf sum

    return window, cluster_emb_sum_dics, cluster_tf_sum_dics, time.time() - start_time


def cluster_outliers(window, cluster_centers, cluster_emb_sum_dics, cluster_tf_sum_dics, min_articles, verbose = False): 
    start_time = time.time()
    out_idx = window[window['cluster'] == -1].index 
    num_new_clusters = int(len(out_idx)/(min_articles))
    
    if num_new_clusters > 1:
        clustering = SphericalKMeans(n_clusters=num_new_clusters).fit(list(window.loc[out_idx, 'embedding'].values))
        cluster_id_dic = {}
        new_centers = []
        new_cluster_id = len(cluster_centers)
        for l in set(clustering.labels_):
            if list(clustering.labels_).count(l) < (min_articles): 
                continue # skip if less than min article number
            cluster_id_dic[l] = new_cluster_id
            new_centers.append(clustering.cluster_centers_[l])
            cluster_emb_sum_dics.append({})
            cluster_tf_sum_dics.append({})
            if verbose: print("A new cluster "+str(new_cluster_id)+" of "+str(list(clustering.labels_).count(l))+" articles is created")
            new_cluster_id = new_cluster_id + 1

        for i in range(len(out_idx)):
            if clustering.labels_[i] in cluster_id_dic:
                cluster_id = cluster_id_dic[clustering.labels_[i]]
                window.at[out_idx[i], 'cluster'] = cluster_id
                article_date = window.loc[out_idx[i], 'date']
                                
                if article_date not in cluster_emb_sum_dics[cluster_id]:
                    cluster_emb_sum_dics[cluster_id][article_date] = [0,0]
                    cluster_tf_sum_dics[cluster_id][article_date] = 0
                cluster_emb_sum_dics[cluster_id][article_date][0] += window.loc[out_idx[i], 'embedding'] #embedding sum
                cluster_emb_sum_dics[cluster_id][article_date][1] += 1 #count of articles
                cluster_tf_sum_dics[cluster_id][article_date] += window.loc[out_idx[i],'article_TF'] #article tf sum
            else:
                window.at[out_idx[i], 'cluster'] = -1
                        
        cluster_centers = np.array(list(cluster_centers) + new_centers)
        
    return window, cluster_centers, cluster_emb_sum_dics, cluster_tf_sum_dics, time.time() - start_time

def update_cluster_keywords_articles(i, window, all_vocab, cluster_keywords_df, cluster_topN_indices):
    for k in cluster_topN_indices.keys(): 
        if k not in cluster_keywords_df.columns:
            cluster_keywords_df[k] = ''
            cluster_keywords_df[k] = cluster_keywords_df[k].astype('object')
        cluster_keywords_df.at[i,k] = ''
        cluster_keywords_df.at[i,k] = [all_vocab[i] for i in cluster_topN_indices[k]]
    return cluster_keywords_df
