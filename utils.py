from __future__ import print_function
import numpy as np
import math, codecs, random, zipfile
from collections import Counter

# return two randomly initialized matrices, one for target words and one for contexts
def init_parameters(dim, vocab_size):
    W = np.random.uniform(-0.1, 0.1, (vocab_size, dim))
    C = np.random.uniform(-0.1, 0.1, (vocab_size, dim))
    return W, C


# return k random context words
# if checking gradients, return a fixed set of words
def negative_sampling(sampling_dist, do_grad_check, k=3):
    if do_grad_check:
        ks = [4,3,2,2]
    else:
        ks = np.random.choice(sampling_dist, size=k)
    return ks


# squish them into one parameter vector (for gradient check)
def roll_params(W, C):
    return np.concatenate([W, C]).ravel()


def unroll_params(rolled_params, dim, vocab_size):
    W = rolled_params[:vocab_size * dim].reshape((vocab_size, dim))
    C = rolled_params[vocab_size * dim:].reshape((vocab_size, dim))
    return W, C


# check if SGNS loss was computed correctly
def obj_check(obj_and_grad, dim, vocab_size, t=3, c=9):

    print('running loss check...')
    test_params = [(0.5, 1, -4.483182), (0.2, 0.6, -3.690505)]
    num_errors = 0
    
    # set some test parameters... 
    for idx, (wmax, cmax, corr_obj) in enumerate(test_params):
        W = np.linspace(-wmax, wmax, num=dim*vocab_size).reshape((vocab_size, dim))
        C = np.linspace(-cmax, cmax, num=dim*vocab_size).reshape((vocab_size, dim))
        L, _, _ = obj_and_grad(t, c, W, C, None, do_grad_check=True)
        if not np.allclose(corr_obj, L, atol=1e-5):
            print('for test case %d, you got %f, while the correct objective should be %f' % (idx, L, corr_obj))
            num_errors += 1

    if num_errors == 0:
        print("congratulations! you've successfully implemented the SGNS objective.")

        
# check if SGNS derivatives were computed correctly
def gradient_check(obj_and_grad, W, C, dim, vocab_size, t=3, c=9):

    params = roll_params(W, C)
    d_params = params.shape[0]
    print('running gradient check...')
    
    cost, dW, dC = obj_and_grad(t, c, W, C, None, do_grad_check=True)
    actual_grad = roll_params(dW, dC)
    num_grad = np.zeros(actual_grad.shape)

    mean = 1e-6 * ( (1 + np.linalg.norm(params)) / d_params)

    # compute per dimension numerical gradients
    for i in range(d_params):
        curr_param = np.zeros( (num_grad.shape))
        curr_param[i] += mean
        new_params = params + curr_param
        new_W, new_C = unroll_params(new_params, dim, vocab_size)
        part_cost, _, _ = obj_and_grad(t, c, new_W, new_C, None, do_grad_check=True)
        num_grad[i] = (part_cost - cost) / mean
        # print i, ' actual: ', num_grad[i], ' mine: ', actual_grad[i]

    num_W, num_C = unroll_params(num_grad, dim, vocab_size)
    num_target_errors = 0
    num_context_errors = 0
    for i in range(vocab_size):
        if not np.allclose(num_W[i], dW[i], atol=1e-5):
            num_target_errors += 1
            print('gradient error at target word index %d' % i)
            print('your calculated gradient', dW[i])
            print('the real gradient', num_W[i])
            print('')

    for i in range(vocab_size):
        if not np.allclose(num_C[i], dC[i], atol=1e-5):
            num_context_errors += 1
            print('gradient error at context word index %d' % i)
            print('your calculated gradient', dC[i])
            print('the real gradient', num_C[i])
            print('')

    if num_target_errors == num_context_errors == 0:
        print("congratulations! you've successfully implemented the SGNS gradient.")
        

# word2vec takes negative samples using the unigram distribution raised to 0.75
def compute_unigram_sampling_dist(vocab, w_to_idx, gamma=0.75):
    vocab_list = []
    for w in vocab:
        weighted_count = math.pow(vocab[w], gamma)
        for i in range(int(weighted_count)):
            vocab_list.append(w_to_idx[w])
    return np.array(vocab_list)


# open the text8 file and compute the vocabulary
def compute_vocab(path_to_data):
    vocab = Counter()
    f = codecs.open(path_to_data, 'r', 'utf-8')
    text = f.read()
    text = text.strip().split()
    for w in text:
        vocab[w] += 1

    w_to_idx = dict(zip(vocab.keys(), range(len(vocab))))
    sampling_dist = compute_unigram_sampling_dist(vocab, w_to_idx)

    return [w_to_idx[w] for w in text], w_to_idx, sampling_dist


# produce next context to train on
def next_context(idxes, window_size):
    contexts = range(0, len(idxes) - window_size, window_size)
    random.shuffle(contexts)
    for i in contexts:
        yield idxes[i:i+window_size]


# given a word, find its nearest neighbors using cosine distance
def nearest_neighbors(word, vocab, idx_to_w, W):

    # normalize W and the query word embedding
    W_norm = W / np.linalg.norm(W, axis=1)[:, None]
    w_v = W_norm[vocab[word]]

    # now compute dot product (cosine distance since we normalized)
    sims = W_norm.dot(w_v)

    # sort by cosine distance and print the 10 highest ones
    ordered_words = np.argsort(sims)[::-1]
    print(word, ':')
    for idx in ordered_words[:10]:
        if idx != vocab[word]: 
            print(idx_to_w[idx], sims[idx])
    print('')


# load pretrained embeddings for the given vocab
def load_embeddings(fname, vocab, idx_to_w):
    vec_file = zipfile.ZipFile('glove.6B.50d.txt.zip', 'r').open('glove.6B.50d.txt', 'r')

    glove_W = {}
    for line in vec_file:
        split = line.strip().split()
        split[0] = split[0].lower().decode('utf-8') # all tokens are lowercased in our dataset
        try:
            tmp = vocab[split[0]]
            glove_W[split[0]] = np.array(split[1:], dtype="float32")
        except:
            pass

    W = np.random.randn(len(vocab), len(glove_W['the']))
    for i in range(len(vocab)):
        w = idx_to_w[i]
        try:
            W[i] = glove_W[w]
        except:
            pass

    return W

