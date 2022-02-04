import re

with open('general-tweets.txt') as file:
    lines = [line.rstrip() for line in file]


def preprocess(text):
    #the function prpocess a single line
    #it takes the single line and returns a list of tokens for that particular line
    #make text lower
    #cleaned_text = text.lower()
    cleaned_text = text
    #remove non-ASCII characters
    #cleaned_text = re.sub(r'[^\x00-\x7F]+',' ', cleaned_text)
    # remove URLS
    cleaned_text = re.sub(r"http\S+", "<URL>", cleaned_text)
    # remove HTs
    cleaned_text = re.sub(r"#[A-Za-z0-9_]+", "<HASHTAG>", cleaned_text)
    # remove Mentions
    cleaned_text = re.sub(r"@[A-Za-z0-9_]+", "<MENTION>", cleaned_text)
    #replace percentage quantities with tags
    cleaned_text = re.sub(r'(\d+(\.\d+)?%)',"<PERCENT>",cleaned_text)
    #replace numbers with tags
    cleaned_text = re.sub("^\d+\s|\s\d+\s|\s\d+$", "<NUM>", cleaned_text)
    #hypenated words are accounted for by joining them/merging them together
    cleaned_text = re.sub(r'\w+(?:-\w+)+', '', cleaned_text)
    # Substitue for punctuations
    cleaned_text = re.sub(r"(\'t)"," not",cleaned_text)
    cleaned_text = re.sub(r'(i\'m)',"i am",cleaned_text)
    cleaned_text = re.sub(r'(ain\'t)',"am not",cleaned_text)
    cleaned_text = re.sub(r'(\'ll)'," will",cleaned_text)
    cleaned_text = re.sub(r'(\'ve)'," have",cleaned_text)
    cleaned_text = re.sub(r'(\'re)'," are",cleaned_text)
    cleaned_text = re.sub(r'(\'s)'," is",cleaned_text)
    cleaned_text = re.sub(r'(\'re)'," are",cleaned_text)
    #removing repetetive spam
    cleaned_text = re.sub('\!\!+', '!', cleaned_text)
    cleaned_text = re.sub('\*\*+', '*', cleaned_text)
    cleaned_text = re.sub('\>\>+', '>', cleaned_text)
    cleaned_text = re.sub('\<\<+', '<', cleaned_text)
    cleaned_text = re.sub('\?\?+', '?', cleaned_text)
    cleaned_text = re.sub('\!\!+', '!', cleaned_text)
    cleaned_text = re.sub('\.\.+', '.', cleaned_text)
    cleaned_text = re.sub('\,\,+', ',', cleaned_text)
    #matching punctuation characters at end of sentences and padding them
    cleaned_text = re.sub('([;:.,!?()])', r' \1 ', cleaned_text)
    #removing multiple spaces finally
    cleaned_text = re.sub('\s{2,}', ' ', cleaned_text)
    #remove trailing white spaces
    cleaned_text = re.sub(r'\s+$', '', cleaned_text) #important to get rid of empty tokens at the end of list
    #tokenization based on spaces for each line
    # spaces = r"\s+"
    # tokenized_sent = re.split(spaces, cleaned_text)
    return cleaned_text

# for i in lines[:5]:
#     print(preprocess(i))

textfile = open("2020114017_tokenize.txt", "w")
for element in lines:
    textfile.write(preprocess(element) + "\n")  
textfile.close()