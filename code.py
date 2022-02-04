import sys

with open('try.txt') as file:
    fdata = file.readlines()

import nltk
from numpy import tri

# Initializing n-gram dictionaries upto n=4.
unigram = {}
bigram = {}
trigram = {}
fourgram = {}

# Function to make n-gram dictionaries with nested sequence of words.


def ngram_dict(data):
    token_sent = []
    for sent in data:
        token_sent.append(nltk.word_tokenize(sent))

    for sent in token_sent:
        for token in sent:
            if token in unigram.keys():
                unigram[token] += 1
            else:
                unigram[token] = 1

    for sent in token_sent:
        for i in range(len(sent)-1):
            if sent[i] not in bigram.keys():
                # If a key isnt present, u make that into a dictionary to work with as well
                bigram[sent[i]] = {}
            if sent[i+1] not in bigram[sent[i]].keys():
                # Now that a bigram dict is also made, u append the same way as u did for unigrams
                bigram[sent[i]][sent[i+1]] = 1
            else:
                bigram[sent[i]][sent[i+1]] += 1

    for sent in token_sent:
        for i in range(len(sent)-2):
            if sent[i] not in trigram.keys():
                # If a key isnt present, u make that into a dictionary to work with as well
                trigram[sent[i]] = {}
            if sent[i+1] not in trigram[sent[i]].keys():
                # Now that a bigram dict is also made, u append the same way as u did for unigrams
                trigram[sent[i]][sent[i+1]] = {}
            if sent[i+2] not in trigram[sent[i]][sent[i+1]].keys():
                trigram[sent[i]][sent[i+1]][sent[i+2]] = 1
            else:
                trigram[sent[i]][sent[i+1]][sent[i+2]] += 1

    for sent in token_sent:
        for i in range(len(sent)-3):
            if sent[i] not in fourgram.keys():
                # If a key isnt present, u make that into a dictionary to work with as well
                fourgram[sent[i]] = {}
            if sent[i+1] not in fourgram[sent[i]].keys():
                # Now that a bigram dict is also made, u append the same way as u did for unigrams
                fourgram[sent[i]][sent[i+1]] = {}
            if sent[i+2] not in fourgram[sent[i]][sent[i+1]].keys():
                fourgram[sent[i]][sent[i+1]][sent[i+2]] = {}
            if sent[i+3] not in fourgram[sent[i]][sent[i+1]][sent[i+2]].keys():
                fourgram[sent[i]][sent[i+1]][sent[i+2]][sent[i+3]] = 1
            else:
                fourgram[sent[i]][sent[i+1]][sent[i+2]][sent[i+3]] += 1


ngram_dict(fdata)
# for item in bigram.items():
#     print(item[0])

# Items is everything, items[1] is the first nested dictionary and u can print their keys and values


def kneyser_ney(n, n_gram, high_ord=True):
    if n == 1:
        d_1 = 0.5
        if high_ord:
            if (unigram[n_gram[0]]) > 0:
                # since it is higher order, discount factor = 0
                numer = unigram[n_gram[0]]
            else:
                numer = 0
            # Total frequency count of the all unigrams
            denom = sum([item[1] for item in unigram.items()])
            return numer/denom  # terms after lambda are zero, so no P_cont here
        else:
            # now we include pcont, so here:
            # num = different string types before the final word
            # denom = the number of different possible n-gram types
            p_cont = len(
                set([item[0] for item in bigram.items() if n_gram[0] in item[1].keys()]))
            denom_cont = len(bigram.keys())
            return p_cont/denom_cont
    if n == 2:
        d_2 = 0.75
        lambda_2 = (d_2 * len(bigram[n_gram[0]]))/(sum([item[1]
                                                        for item in bigram[n_gram[0]].items()]))
        if high_ord:
            if n_gram[1] in bigram[n_gram[0]].keys():
                num_2 = bigram[n_gram[0]][n_gram[1]]
            else:
                num_2 = 0
            # total frequency count
            return ((max(num_2-d_2, 0))/sum([item[1] for item in bigram[n_gram[0]].items()])) + (lambda_2*kneyser_ney(1, n_gram[1:], False))
        else:
            num_2 = 0
            for first in trigram.keys():
                if n_gram[0] in trigram[first].keys():
                    if n_gram[1] in trigram[first][n_gram[0]].keys():
                        num_2 += 1
            return (max(num_2-d_2, 0)/len(trigram.keys()))+(lambda_2*kneyser_ney(1, n_gram[1:], False))
    # if n == 3:
    #     d_3 = 0.75
    #     lambda_3 = (d_3 * len(trigram[n_gram[0]][n_gram[1]]))/(sum([item[1] for item in trigram[n_gram[0]][n_gram[1]].items()]))
    #     if high_ord:

    # if n == 4:
    #     d_4 = 0.75
    #     lambda_4 = (d_4 * len(fourgram[n_gram[0]][n_gram[1]][n_gram[2]]))/(sum([item[1] for item in fourgram[n_gram[0]][n_gram[1]][n_gram[2]].items()]))


if __name__ == "__main__":
    #check if all arguments are given
    if len(sys.argv) != 4:
        print("please provide all the arguments")
        exit()
    #obtain n from the first argument
    n = int(sys.argv[1])
    #check for w and k from the second argument
    if sys.argv[2] == 'w':
        sm_model = kneyser_ney
    elif sys.argv[2] == 'k':
        smoothing = kneyser_ney
    else:
        print("Incorrect Model Option, Model NOT defined")

    print("Training on corpus")
    ngram_dict(sys.argv[3])
    print("Training complete")
    sentence = input("Input sentence: ")
    sentence = [str(tokens).lower() for tokens in tokenizer(sentence)]
    for i, token in enumerate(sentence):
        sentence[i] = token if token in unigram.keys() else "<unk>"
    length = len(sentence)
    for i in range(n-1):
        sentence.insert(0, '<start>')
    ans = 1
    for i in range(length):
        prob = smoothing(n, sentence[i:i+n])
        print("for", sentence[i:i+n], ':', prob)
        ans = ans * prob
    print("Final output:", ans)
