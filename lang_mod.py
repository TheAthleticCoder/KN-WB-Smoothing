import nltk
from nltk.tokenize import word_tokenize
import sys
import re
import math


# Initializing n-gram dictionaries upto n=4.
unigram = {}
bigram = {}
trigram = {}
fourgram = {}


def preprocess(text):
    # the function prpocess a single line
    # it takes the single line and returns a list of tokens for that particular line
    # make text lower
    cleaned_text = text.lower()    
    # remove non-ASCII characters
    #cleaned_text = re.sub(r'[^\x00-\x7F]+',' ', cleaned_text)
    # remove URLS
    cleaned_text = re.sub(r"http\S+", "<URL>", cleaned_text)
    # remove HTs
    cleaned_text = re.sub(r"#[A-Za-z0-9_]+", "<HASHTAG>", cleaned_text)
    # remove Mentions
    cleaned_text = re.sub(r"@[A-Za-z0-9_]+", "<MENTION>", cleaned_text)
    # replace percentage quantities with tags
    cleaned_text = re.sub(r'(\d+(\.\d+)?%)', "<PERCENT>", cleaned_text)
    # replace numbers with tags
    cleaned_text = re.sub("^\d+\s|\s\d+\s|\s\d+$", " <NUM> ", cleaned_text)
    # cleaned_text = re.sub(r'[0-9]', " <NUM> ", cleaned_text)
    # hypenated words are accounted for by joining them/merging them together
    cleaned_text = re.sub(r'\w+(?:-\w+)+', '', cleaned_text)
    # Substitue for punctuations
    cleaned_text = re.sub(r"(\'t)", " not", cleaned_text)
    cleaned_text = re.sub(r'(i\'m)', "i am", cleaned_text)
    cleaned_text = re.sub(r'(ain\'t)', "am not", cleaned_text)
    cleaned_text = re.sub(r'(\'ll)', " will", cleaned_text)
    cleaned_text = re.sub(r'(\'ve)', " have", cleaned_text)
    cleaned_text = re.sub(r'(\'re)', " are", cleaned_text)
    cleaned_text = re.sub(r'(\'s)', " is", cleaned_text)
    cleaned_text = re.sub(r'(\'re)', " are", cleaned_text)
    # removing repetetive spam
    cleaned_text = re.sub('\!\!+', '!', cleaned_text)
    cleaned_text = re.sub('\*\*+', '*', cleaned_text)
    cleaned_text = re.sub('\>\>+', '>', cleaned_text)
    cleaned_text = re.sub('\<\<+', '<', cleaned_text)
    cleaned_text = re.sub('\?\?+', '?', cleaned_text)
    cleaned_text = re.sub('\!\!+', '!', cleaned_text)
    cleaned_text = re.sub('\.\.+', '.', cleaned_text)
    cleaned_text = re.sub('\,\,+', ',', cleaned_text)
    cleaned_text = re.sub('\:\:+', ':', cleaned_text)
    cleaned_text = re.sub('\;\;+', ';', cleaned_text)
    # matching punctuation characters at end of sentences and padding them
    cleaned_text = re.sub('([;:.,!?()])', r' \1 ', cleaned_text)
    # removing multiple spaces finally
    cleaned_text = re.sub('\s{2,}', ' ', cleaned_text)
    # remove trailing white spaces
    # important to get rid of empty tokens at the end of list
    cleaned_text = re.sub(r'\s+$', '', cleaned_text)
    # tokenization based on spaces for each line
    spaces = r"\s+"
    tokenized_sent = re.split(spaces, cleaned_text)
    return tokenized_sent

def ngram_dict(data):
    token_sent = []
    for sent in data:
        token_sent.append(preprocess(sent))

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
            p_cont = len(set([item[0] for item in bigram.items() if n_gram[0] in item[1].keys()]))
            denom_cont = len(bigram.keys())
            return p_cont/denom_cont
    if n == 2:
        d_2 = 0.75
        lambda_2 = (d_2 * len(bigram[n_gram[0]]))/(sum([item[1] for item in bigram[n_gram[0]].items()]))
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
    if n == 3:
        d_3 = 0.75
        lambda_3 = (d_3 * len(trigram[n_gram[0]][n_gram[1]]))/(sum([item[1] for item in trigram[n_gram[0]][n_gram[1]].items()]))
        if high_ord:
            if n_gram[1] in trigram[n_gram[0]].keys():
                if n_gram[2] in trigram[n_gram[0]][n_gram[1]].keys():
                    num_3 = trigram[n_gram[0]][n_gram[1]][n_gram[2]]
                else:
                    num_3 = 0
            else:
                num_3 = 0
            # total frequency count
            return ((max(num_3-d_3, 0))/sum([item[1] for item in trigram[n_gram[0]][n_gram[1]].items()])) + (lambda_3*kneyser_ney(2, n_gram[1:], False))
        else:
            num_3 = 0
            for first in fourgram.keys():
                if n_gram[0] in fourgram[first].keys():
                    if n_gram[1] in fourgram[first][n_gram[0]].keys():
                        if n_gram[2] in fourgram[first][n_gram[0]][n_gram[1]].keys():
                            num_3 += 1
            return (max(num_3-d_3, 0)/len(fourgram.keys()))+(lambda_3*kneyser_ney(2, n_gram[1:], False))
    if n == 4:
        d_4 = 0.75
        lambda_4 = (d_4 * len(fourgram[n_gram[0]][n_gram[1]][n_gram[2]]))/(sum([item[1] for item in fourgram[n_gram[0]][n_gram[1]][n_gram[2]].items()]))
        if high_ord:
            if n_gram[1] in fourgram[n_gram[0]].keys():
                if n_gram[2] in fourgram[n_gram[0]][n_gram[1]].keys():
                    if n_gram[3] in fourgram[n_gram[0]][n_gram[1]][n_gram[2]].keys():
                        num_4 = fourgram[n_gram[0]][n_gram[1]][n_gram[2]][n_gram[3]]
                    else:
                        num_4 = 0
                else:
                    num_4 = 0
            else:
                num_4 = 0
            # total frequency count
            return ((max(num_4-d_4, 0))/sum([item[1] for item in fourgram[n_gram[0]][n_gram[1]][n_gram[2]].items()])) + (lambda_4*kneyser_ney(3, n_gram[1:], False))
        else:
            num_4 = 0
            for first in fourgram.keys():
                if n_gram[0] in fourgram[first].keys():
                    if n_gram[1] in fourgram[first][n_gram[0]].keys():
                        if n_gram[2] in fourgram[first][n_gram[0]][n_gram[1]].keys():
                            if n_gram[3] in fourgram[first][n_gram[0]][n_gram[1]][n_gram[2]].keys():
                                num_4 += 1
            return (max(num_4-d_4, 0)/len(fourgram.keys()))+(lambda_4*kneyser_ney(3, n_gram[1:], False))

    # if n == 3:
    #     d_3 = 0.75
    #     lambda_3 = (d_3 * len(trigram[n_gram[0]][n_gram[1]]))/(sum([item[1] for item in trigram[n_gram[0]][n_gram[1]].items()]))
    #     if high_ord:

    # if n == 4:
    #     d_4 = 0.75
    #     lambda_4 = (d_4 * len(fourgram[n_gram[0]][n_gram[1]][n_gram[2]]))/(sum([item[1] for item in fourgram[n_gram[0]][n_gram[1]][n_gram[2]].items()]))


#calculating perplexity using an array of probabilities
def perplexity(prob_array):
    log_prob = 0
    for i in range(len(prob_array)):
        log_prob += math.log(prob_array[i])
    return math.exp(-log_prob/len(prob_array))


# check if all arguments are given
if len(sys.argv) != 4:
    print("please provide all the arguments")
    exit()
# obtain n, model type and path from the first argument
num = int(sys.argv[1])
model = sys.argv[2]
path = sys.argv[3]

if num > 4:
    n = 4
else:
    n = num

if model == "k":
    smooth_model = kneyser_ney
elif model == "w":
    smooth_model = witten_bell
else:
    print("Please provide a valid model type")
    exit()

# reading file and storing the data
with open(path) as file:
    fdata = file.readlines()

# generating the uni,bi,tri,four gram dictionaries
ngram_dict(fdata)
print("Trained")
sentence = input("input sentence: ")
sentence = preprocess(sentence)

perplex = []
for i in range(len(sentence)-n):
    ans = 1
    prob = kneyser_ney(n,sentence[i:i+n])
    print("for",sentence[i:i+n],':',prob)
    ans = ans * prob
    perplex.append(ans)
print("Final perplexity:", perplexity(perplex))



