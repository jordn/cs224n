#!/usr/bin/env python

import random

import numpy as np
from q1_softmax import softmax
from q2_gradcheck import gradcheck_naive
from q2_sigmoid import sigmoid


def normalizeRows(x):
    """Row normalization function.

    Implement a function that normalizes each row of a matrix to have
    unit length.
    """

    ### YOUR CODE HERE
    return x / np.sqrt(np.sum(np.square(x), axis=1, keepdims=True))
    ### END YOUR CODE


def test_normalize_rows():
    print("Testing normalizeRows...")
    x = normalizeRows(np.array([[3.0, 4.0], [1, 2]]))
    print(x)
    ans = np.array([[0.6, 0.8], [0.4472136, 0.89442719]])
    assert np.allclose(x, ans, rtol=1e-05, atol=1e-06)
    print("")


def softmax_cost_and_gradient(predicted, target, output_vectors, dataset):
    """Softmax cost function for word2vec models.

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, assuming the softmax prediction function and cross
    entropy loss.

    Arguments:
    predicted -- numpy ndarray, predicted word vector (\hat{v} in
                 the written component)
    target -- integer, the index of the target word
    output_vectors -- "output" vectors (as rows) for all tokens
    dataset -- needed for negative sampling, unused here.

    Return:
    cost -- cross entropy cost for the softmax word prediction
    grad_pred -- the gradient with respect to the predicted word
           vector
    grad -- the gradient with respect to all the other word
           vectors

    We will not provide starter code for this function, but feel
    free to reference the code you previously wrote for this
    assignment!
    """

    ### YOUR CODE HERE
    scores = predicted @ output_vectors.T
    probabilities = softmax(scores)
    # We wish to minimise the cross entropy between the target
    # and actual distributions.
    cost = -np.log(probabilities[target])

    a2 = predicted
    W2 = output_vectors
    z2 = scores
    y = probabilities

    grad_z2 = y
    grad_z2[target] -= 1

    grad_W2 = grad_z2.reshape(-1, 1) @ a2.reshape(1, -1)
    grad_a2 = W2.T @ grad_z2
    grad_z1 = grad_a2  # Linear activation
    # grad_W1 = a1 @ grad_z1.reshape(1, -1)
    grad_in = grad_z1 * 1  # grad_w1 for the active column

    grad = grad_W2
    grad_pred = grad_z1

    ### END YOUR CODE

    return cost, grad_pred, grad


def get_negative_samples(target, dataset, K):
    """ Samples K indexes which are not the target """

    indices = [None] * K
    for k in range(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == target:
            newidx = dataset.sampleTokenIdx()
        indices[k] = newidx
    return indices


def neg_sampling_cost_and_gradient(predicted, target, output_vectors, dataset, K=10):
    """Negative sampling cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, using the negative sampling technique. K is the sample
    size.

    Note: See test_word2vec below for dataset's initialization.

    Arguments/Return Specifications: same as softmaxCostAndGradient
    """
    # Sampling of indices is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    indices = [target]
    indices.extend(get_negative_samples(target, dataset, K))

    ### YOUR CODE HERE
    sampled_tokens = indices[1:]

    # The probabilities are unnormalized as we don't softmax them.
    # But the gradient is still in the right direction.
    z2 = predicted @ output_vectors[indices].T
    scores = sigmoid(z2)

    # "Probability" in inverted commas as this is not normalised.
    prob_target = scores[0]
    prob_negs = scores[1:]  # K "probabbilities" of neg samples.

    # Instead of minimising the cross entropy, we wish to maximise
    # the log probability of the target word and minimise the probability
    # of the neg samples. Equivalently, minimise the (neg log likelihood of the data
    # + neg log likelihood of not the fake data) as we do here.
    cost = - (np.log(prob_target) + sum(np.log(1 - prob_negs)))

    grad_out = np.zeros_like(output_vectors)
    for i, token in enumerate(indices):
        grad_out[token] += scores[i] * predicted
    grad_out[target] += -predicted

    grad_pred = ((scores[0] - 1) * output_vectors[target]
                 + scores[1:] @ output_vectors[sampled_tokens])

    ### END YOUR CODE

    return cost, grad_pred, grad_out


def skipgram(current_word, context_size, context_words, tokens, input_vectors, output_vectors,
             dataset, word2vec_cost_and_gradient=softmax_cost_and_gradient):
    """Skip-gram model in word2vec.

    Implement the skip-gram model in this function.

    Arguments:
    current_word -- a string of the current center word
    context_size -- integer, context size
    context_words -- list of no more than 2*C strings, the context words
    tokens -- a dictionary that maps words to their indices in
              the word vector list
    input_vectors -- "input" word vectors (as rows) for all tokens
    output_vectors -- "output" word vectors (as rows) for all tokens
    word2vec_cost_and_gradient -- the cost and gradient function for
                               a prediction vector given the target
                               word vectors, could be one of the two
                               cost functions you implemented above.

    Return:
    cost -- the cost function value for the skip-gram model
    grad -- the gradient with respect to the word vectors
    """

    cost = 0.0
    grad_in = np.zeros(input_vectors.shape)
    grad_out = np.zeros(output_vectors.shape)

    ### YOUR CODE HERE

    # We use the the input word to "predict" the output words
    # (through their respective mappings).
    input_embedding = input_vectors[tokens[current_word]]  # == z1 == a1 = "h"

    # # Unnecessary block viewing it as a neural net.
    # a1 = np.zeros((len(tokens)))
    # a1[tokens[current_word]] = 1
    # W1 = input_vectors
    # z1 = W1.T @ a1  # Equivalent to input_embedding above.
    # a2 = z1  # Hidden layer has linear activation.
    # W2 = output_vectors
    # z2 = W2 @ a2  # Equivalent np.dot(W2, a2)
    # y = softmax(z2)

    for context_word in context_words:
        target = tokens[context_word]
        _cost, _grad_pred, _grad_out = word2vec_cost_and_gradient(input_embedding, target,
                                                                  output_vectors, dataset)

        cost += _cost
        grad_out += _grad_out
        grad_in[tokens[current_word]] += _grad_pred

    ### END YOUR CODE

    return cost, grad_in, grad_out


def cbow(current_word, C, context_words, tokens, input_vectors, output_vectors,
         dataset, word2vec_cost_and_gradient=softmax_cost_and_gradient):
    """CBOW model in word2vec

    Implement the continuous bag-of-words model in this function.

    Arguments/Return specifications: same as the skip-gram model

    Extra credit: Implementing CBOW is optional, but the gradient
    derivations are not. If you decide not to implement CBOW, remove
    the NotImplementedError.
    """

    cost = 0.0
    grad_in = np.zeros(input_vectors.shape)
    grad_out = np.zeros(output_vectors.shape)

    ### YOUR CODE HERE

    # We use a smush of the context words to "predict" the center word.
    context_embeddings = input_vectors[[tokens[w] for w in context_words]]  # <10x3>
    input_embedding = np.sum(context_embeddings, axis=0)  # <1x3>

    # # Unnecessary block just viewing it as a neural net.
    # a1 = np.zeros((len(tokens)))
    # for w in context_words:
    #     a1[tokens[w]] += 1
    # W1 = input_vectors
    # z1 = W1.T @ a1  # Equivalent to input_embedding above.
    # a2 = z1  # Hidden layer has linear activation.
    # W2 = output_vectors
    # z2 = W2 @ a2  # Equivalent np.dot(W2, a2)
    # y = softmax(z2)

    target = tokens[current_word]
    cost, _grad_pred, grad_out = word2vec_cost_and_gradient(input_embedding, target,
                                                            output_vectors, dataset)
    for w in context_words:
        grad_in[tokens[w]] += _grad_pred
    ### END YOUR CODE

    return cost, grad_in, grad_out


#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, tokens, word_vectors, dataset, max_context_size,
                         word2vecCostAndGradient=softmax_cost_and_gradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(word_vectors.shape)
    num_tokens = word_vectors.shape[0] // 2  #
    input_vectors = word_vectors[:num_tokens, :]
    output_vectors = word_vectors[num_tokens:, :]

    for i in range(batchsize):
        # Randomly sample a center token and a random length of context tokens.
        context_size = random.randint(1, max_context_size)
        centerword, context = dataset.getRandomContext(context_size)

        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1  ## ??

        batch_cost, grad_in, grad_out = word2vecModel(
            centerword, context_size, context, tokens, input_vectors, output_vectors,
            dataset, word2vecCostAndGradient)
        cost += batch_cost / batchsize / denom
        grad[:num_tokens, :] += grad_in / batchsize / denom
        grad[num_tokens:, :] += grad_out / batchsize / denom

    return cost, grad


def test_word2vec():
    """ Interface to the dataset for negative sampling """
    dataset = type('dummy', (), {})()

    tokens = ["a", "b", "c", "d", "e"]
    num_tokens = len(tokens)
    dims = 3
    C = 5

    def dummySampleTokenIdx():
        return random.randint(0, num_tokens - 1)

    def getRandomContext(C):
        return (tokens[random.randint(0, num_tokens - 1)],
                [tokens[random.randint(0, num_tokens - 1)] for _ in range(2 * C)])

    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    dummy_vectors = normalizeRows(np.random.randn(2 * num_tokens, dims))
    dummy_tokens = dict([(token, index) for index, token in enumerate(tokens)])

    print()
    print("==== Gradient check for skip-gram ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, C, softmax_cost_and_gradient
    ), dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, neg_sampling_cost_and_gradient),
                    dummy_vectors)
    print("\n==== Gradient check for CBOW      ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        cbow, dummy_tokens, vec, dataset, 5, softmax_cost_and_gradient),
                    dummy_vectors)

    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        cbow, dummy_tokens, vec, dataset, 5, neg_sampling_cost_and_gradient),
                    dummy_vectors)

    print("\n=== Results ===")
    print(skipgram("c", 3, ["a", "b", "e", "d", "b", "c"],
                   dummy_tokens, dummy_vectors[:5, :], dummy_vectors[5:, :], dataset))
    print(skipgram("c", 1, ["a", "b"],
                   dummy_tokens, dummy_vectors[:5, :], dummy_vectors[5:, :], dataset,
                   neg_sampling_cost_and_gradient))
    print(cbow("a", 2, ["a", "b", "c", "a"],
               dummy_tokens, dummy_vectors[:5, :], dummy_vectors[5:, :], dataset))
    print(cbow("a", 2, ["a", "b", "a", "c"],
               dummy_tokens, dummy_vectors[:5, :], dummy_vectors[5:, :], dataset,
               neg_sampling_cost_and_gradient))


if __name__ == "__main__":
    random.seed(31415)
    np.random.seed(9265)
    test_normalize_rows()
    test_word2vec()
