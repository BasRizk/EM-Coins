# -*- coding: utf-8 -*-
import requests, json
import numpy as np

from sklearn.mixture import GaussianMixture
# from sklearn.cluster import KMeans
from scipy.stats import binom

from concurrent.futures import ThreadPoolExecutor


# =============================================================================
# Retrieving Dataset using a generator API
# =============================================================================
def get_dataset(num_of_samples=30):

    with ThreadPoolExecutor(max_workers=num_of_samples) as pool:
        urls = ['https://24zl01u3ff.execute-api.us-west-1.amazonaws.com/beta']*num_of_samples
        response_list = list(pool.map(lambda url: requests.get(url),urls))
        
    dataset = np.array(list(map(lambda x: json.loads(x.json()['body']), response_list)))
    dataset = dataset.sum(axis=1)/len(dataset[0])
    return dataset.reshape(num_of_samples, 1)

# =============================================================================
# Retrieving sample Dataset from 
# https://ra-training.s3-us-west-1.amazonaws.com/DoBatzoglou_2008_EMAlgo.pdf
# =============================================================================
def get_paper_sample():
    return np.array([0.5, 0.9, 0.8, 0.4, 0.7]).reshape(-1, 1)

# =============================================================================
# Retrieving a local sample
# =============================================================================
def get_local_sample(a=0.35, b=0.7, num_of_samples=30, sample_size=20):
    dataset = np.zeros((num_of_samples, 1))
    for i in range(num_of_samples):
        theta = a if np.random.binomial(1, 0.5) == 1 else b
        dataset[i] = np.sum(np.random.binomial(n=1, p=theta, size=sample_size))/sample_size
    return dataset

# =============================================================================
# Gaussian Mixture Model
# =============================================================================
def gmm(dataset, n_init=1,
              init_means=None,
              iter_max=100):
    # init_means=np.array([0.5, 0.5]).reshape((2,1))
    model = GaussianMixture(n_components=2, init_params='kmeans',
                            means_init=init_means,
                            n_init = n_init,
                            max_iter = iter_max)
    
    model.fit(dataset)

    return model, np.sort(model.means_.flatten())

# =============================================================================
# Expectation Maximization Basic -- Specifically implemented for this example
# Adapted from http://karlrosaen.com/ml/notebooks/em-coin-flips/
# =============================================================================
def em(dataset, sample_size, n_iter, init_thetas=None):
    def em_iter(dataset, sample_size, thetas):
        # E-STEP
        count = np.zeros((len(dataset), 2, 2, 1))
        for i, ratio in enumerate(dataset.flatten()):
            h = ratio*sample_size
            t = sample_size-h
            bi_a = binom.pmf(k=h, n=sample_size, p=thetas[0])
            bi_b = binom.pmf(k=h, n=sample_size, p=thetas[1])
            p_e_a = bi_a / (bi_a + bi_b)
            p_e_b = bi_b / (bi_a + bi_b)
            count[i][0][0] = p_e_a*h
            count[i][0][1] = p_e_b*h
            count[i][1][0] = p_e_a*t
            count[i][1][1] = p_e_b*t
        count = count.sum(axis=0)    
        
        # M-STEP
        thetas[0] = count[0][0]/(count[0][0] + count[1][0])
        thetas[1] = count[0][1]/(count[0][1] + count[1][1])
        return thetas

    thetas =  np.random.random(2)
    if init_thetas is not None:
        thetas = init_thetas.copy()
        
    for i in range(n_iter):
        # print('iteration', i, 'thetas:', thetas.round(2))
        thetas = em_iter(dataset, sample_size, thetas)
    # print('iteration', i, 'thetas:', thetas.round(2))
    
    return thetas
        

# sample_size=20
# dataset = get_local_sample(a=0.6, b=0.7, num_of_samples=30, sample_size=20)
# # dataset = get_paper_sample()
# model, means = gmm(dataset, n_init=1, iter_max=3)
# print('means', means)

# thetas_init = None
# thetas_init = np.array([0.6, 0.5])
# basic_em_means = em(dataset, sample_size, n_iter=3, init_thetas=thetas_init)
# print('em', basic_em_means.round(2))

# kmeans = KMeans(n_clusters=2, random_state=0).fit(dataset)
# print('kmeans', np.sort(kmeans.cluster_centers_.flatten()))


# Verification -- Simply calculating max-likelihood based on perdiction
# num_of_samples = len(dataset)
# sample_size = 20
# samples_count = np.zeros((2, num_of_samples, 2, 1))
# for i, (pred, sample) in enumerate(zip(model.predict(dataset), dataset)):
#     samples_count[pred][i] = np.array([sample*sample_size, (1-sample)*sample_size])

# samples_count = samples_count.sum(axis=1)
# theta_a = samples_count[0][0]/(samples_count[0][0] + samples_count[0][1])
# theta_b = samples_count[1][0]/(samples_count[1][0] + samples_count[1][1])
# print(model.means_)
# print(theta_a, theta_b)