#Importing the required libraries
import nltk
import random
from nltk.tokenize import word_tokenize
import sys
import re
import math


# Initializing n-gram dictionaries upto n=4.
unigram = {}
bigram = {}
trigram = {}
fourgram = {}

# Cleaning up text suitably and tokenizing it
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

#how to randomly choose from a list
def random_choice(list):
    return random.choice(list)

def convert_some_to_unk(tokenized_sentences):
    vocab = {}
    for sent in tokenized_sentences:
        for token in sent:
            if token in vocab.keys():
                vocab[token] += 1
            else:
                vocab[token] = 1
    #to_be_changed = int(ratio * len(vocab))
    vocab = {k: v for k, v in sorted(vocab.items(), key=lambda item: item[1])}
    to_be_changed = 0
    for key in vocab.keys():
        if vocab[key] == 1:
            to_be_changed += 1
    words_to_be_changed = list(vocab.keys())[0:to_be_changed]
    # temp_sent = []
    # temp_tokenized_sents = []
    # for sent in tokenized_sentences:
    #     for word in sent:
    #         if word in words_to_be_changed:
    #             temp_sent.append("<UNK>")
    #         else:
    #             temp_sent.append(word)
    #     temp_tokenized_sents.append(temp_sent)
    # tokenized_sentences = temp_tokenized_sents
    # return tokenized_sentences
    for i,sent in enumerate(tokenized_sentences):
        for j,token in enumerate(sent):
            if token in words_to_be_changed:
                tokenized_sentences[i][j] = '<UNK>'
    return tokenized_sentences


#Creating n-gram dictionaries for each n
def ngram_dict(data):
    token_sent = []
    for sent in data:
        token_sent.append(preprocess(sent))
    token_sent = convert_some_to_unk(token_sent)

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

#Implementing Kneyser-Ney for n-grams
## Implementing for different discount values and accounting for high order as well
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
    if n==3:
        d_3 = 0.9
        try:
            lambd = (d_3 * len(trigram[n_gram[0]][n_gram[1]]))/(sum([item[1] for item in trigram[n_gram[0]][n_gram[1]].items()]))
        except:
            return d_3*kneyser_ney(2,n_gram[1:],False)
        count = trigram[n_gram[0]][n_gram[1]][n_gram[2]] if n_gram[2] in trigram[n_gram[0]][n_gram[1]].keys() else 0
        among = sum([item[1] for item in trigram[n_gram[0]][n_gram[1]].items()])
        return (max(0,count-d_3)/among)+(lambd*kneyser_ney(2,n_gram[1:],False))
    if n==4:
        d_4 = 0.9
        try:
            lambd = (d_4 * len(fourgram[n_gram[0]][n_gram[1]][n_gram[2]]))/(sum([item[1] for item in fourgram[n_gram[0]][n_gram[1]][n_gram[2]].items()]))
        except:
            return d_4*kneyser_ney(3,n_gram[1:],False)
        count = fourgram[n_gram[0]][n_gram[1]][n_gram[2]][n_gram[3]] if n_gram[3] in fourgram[n_gram[0]][n_gram[1]][n_gram[2]].keys() else 0
        among = sum([item[1] for item in fourgram[n_gram[0]][n_gram[1]][n_gram[2]].items()])
        return (max(0,count-d_4)/among)+(lambd*kneyser_ney(3,n_gram[1:],False))


#Implementing Witten Bell algorithm
def witten_bell(n, n_gram, high_ord= True):
    if n == 1:
        #We calculate the standard probility of the n_gram, called "original prob"
        num = unigram[n_gram[0]]
        denom = sum([item[1] for item in unigram.items()])
        return (num/denom)
    if n == 2:
        no_tokens = len(bigram[n_gram[0]])
        types = sum([item[1] for item in bigram[n_gram[0]].items()])
        plus_lambda = (types/(no_tokens+types))
        minus_lambda = (no_tokens/(no_tokens+types))
        if n_gram[1] in bigram[n_gram[0]].keys(): #if seen in higher order model, we can use it
            return (plus_lambda)*(bigram[n_gram[0]][n_gram[1]]/types) + (minus_lambda)*(witten_bell(1, n_gram[1:]))
        else:
            return minus_lambda*witten_bell(1, n_gram[1:])
    if n == 3:
        # if n_gram[1] in trigram[n_gram[0]].keys():
        try:
            no_tokens = len(trigram[n_gram[0]][n_gram[1]])
            types = sum([item[1] for item in trigram[n_gram[0]][n_gram[1]].items()])
            plus_lambda = (types/(no_tokens+types))
            minus_lambda = (no_tokens/(no_tokens+types))
            if n_gram[2] in trigram[n_gram[0]][n_gram[1]].keys():
                return (plus_lambda)*(trigram[n_gram[0]][n_gram[1]][n_gram[2]]/types) + (minus_lambda)*(witten_bell(2, n_gram[1:]))
            else:
                return minus_lambda*witten_bell(2, n_gram[1:])
        # else:
        except:
            return 0.9*witten_bell(2, n_gram[1:])
    if n == 4:
        # if n_gram[1] in fourgram[n_gram[0]].keys():
        try:
            try:
                # if n_gram[2] in fourgram[n_gram[0]][n_gram[1]].keys():
                    no_tokens = len(fourgram[n_gram[0]][n_gram[1]][n_gram[2]])
                    types = sum([item[1] for item in fourgram[n_gram[0]][n_gram[1]][n_gram[2]].items()])
                    plus_lambda = (types/(no_tokens+types))
                    minus_lambda = (no_tokens/(no_tokens+types))
                    if n_gram[3] in fourgram[n_gram[0]][n_gram[1]][n_gram[2]].keys():
                        return (plus_lambda)*(fourgram[n_gram[0]][n_gram[1]][n_gram[2]][n_gram[3]]/types) + (minus_lambda)*(witten_bell(3, n_gram[1:]))
                    else:
                        return minus_lambda*witten_bell(3, n_gram[1:])
            except KeyError:
                return 0.9*witten_bell(3, n_gram[1:])
        # else:
        except KeyError:
            return 0.9*witten_bell(3, n_gram[1:])
    

#calculating perplexity using an array of probabilities
def perplexity(prob_array):
    log_prob = 0
    for i in range(len(prob_array)):
        log_prob += math.log(prob_array[i])
    return math.exp(-log_prob/len(prob_array))


### INTERACTIVE PART OF CODE ###
# check if all arguments are given
if len(sys.argv) != 4:
    print("please provide all the arguments")
    exit()

# obtain n, model type and path from the first argument
num = int(sys.argv[1])
model = sys.argv[2]
path = sys.argv[3]

# Making sure that n is valid else we set it to 4(default)
if num > 4:
    n = 4
else:
    n = num

# Choosing the model type based on the input
if model == "k":
    smooth_model = kneyser_ney
elif model == "w":
    smooth_model = witten_bell
else:
    print("Please provide a valid model type")
    exit()

# # reading file and storing the data
# with open(path) as file:
#     fdata = file.readlines()

#####################################################################333333333333333##########
#                        TRAIN - TEST CODE
#####################################################################333333333333333##########
#shuffling data in a list
# with open(path) as file:
#     fdata = file.readlines()
# random.shuffle(fdata)
# # Choosing training and testing data accordingly
# training_data = fdata[:len(fdata)-1000]
# testing_data = fdata[len(fdata)-1000:]
# ngram_dict(training_data)
# final_per = []
# final_sent = []
# count = 0
# print("Training dict is ready")
# for sentence in training_data:
#     print(sentence)
#     print(count)
#     count += 1
#     real_sent =  re.sub(r'\n', '', sentence)
#     sentence = preprocess(sentence)
#     if len(sentence) < 4:
#         for i in range(4-len(sentence)):
#             sentence.append("<UNK>")
#     sen_temp = []
#     for i in sentence:
#         if i in unigram.keys():
#             sen_temp.append(i)
#         else:
#             sen_temp.append("<UNK>")
#     sentence = sen_temp
#     perplex = []
#     for i in range(len(sentence)-n+1):
#         prob = smooth_model(n,sentence[i:i+n])
#         perplex.append(prob)
#     # print("Final perplexity:", perplexity(perplex))
#     final_sent.append(real_sent)
#     final_per.append(perplexity(perplex))
# #python code to find average of a list
# def mean(numbers):
#     return float(sum(numbers)) / max(len(numbers), 1)

# file1 = open("2020114017_LM1_train-perplexity.txt","w")#write mode
# file1.write(str(mean(final_per)) + "\n")
# for i in range(len(final_sent)):
#     file1.write(final_sent[i]+'\t'+str(final_per[i])+'\n')
# file1.close()




# # testing the model
# final_per = []
# final_sent = []
# count = 0
# for sentence in testing_data:
#     print(sentence)
#     print(count)
#     count += 1
#     real_sent =  re.sub(r'\n', '', sentence)
#     sentence = preprocess(sentence)
#     if len(sentence) < 4:
#         for i in range(4-len(sentence)):
#             sentence.append("<UNK>")
#     sen_temp = []
#     for i in sentence:
#         if i in unigram.keys():
#             sen_temp.append(i)
#         else:
#             sen_temp.append("<UNK>")
#     sentence = sen_temp
#     perplex = []
#     for i in range(len(sentence)-n+1):
#         prob = smooth_model(n,sentence[i:i+n])
#         perplex.append(prob)
#     # print("Final perplexity:", perplexity(perplex))
#     final_sent.append(real_sent)
#     final_per.append(perplexity(perplex))
# #python code to find average of a list
# def mean(numbers):
#     return float(sum(numbers)) / max(len(numbers), 1)

# file1 = open("t1.txt","w")#write mode
# file1.write(str(mean(final_per)) + "\n")
# for i in range(len(final_sent)):
#     file1.write(final_sent[i]+'\t'+str(final_per[i])+'\n')
# file1.close()
# print("Over")
    

#####################################################################333333333333333##########
#                                SINGLE SENTENCE
#####################################################################333333333333333##########

# # generating the uni,bi,tri,four gram dictionaries
with open(path) as file:
    fdata = file.readlines()
ngram_dict(fdata)
print("Trained")

sentence = input("input sentence: ")
sentence = preprocess(sentence)
sen_temp = []
for i in sentence:
    if i in unigram.keys():
        sen_temp.append(i)
    else:
        sen_temp.append("<UNK>")
sentence = sen_temp

#Calculating perplexity of the Input sentence.
perplex = []
prob_final = 1
for i in range(len(sentence)-n+1):
    prob = smooth_model(n,sentence[i:i+n])
    print("For",sentence[i:i+n],'probability is:',prob)
    perplex.append(prob)
    prob_final *= prob
print("Final Probability:", prob_final)
print("Final Perplexity:", perplexity(perplex))
