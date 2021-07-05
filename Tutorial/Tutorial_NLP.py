# Original = https://pub.towardsai.net/natural-language-processing-nlp-with-python-tutorial-for-beginners-1f54e610a1a0

# READING FILE
text_file = open('Simple.txt')
text = text_file.read()
text

# IMPORTING NLTK FILE
import nltk
nltk.download()
nltk.download('punkt')
from nltk import sent_tokenize
from nltk import word_tokenize

# SENT TOKENIZE
Sentences = sent_tokenize(text)
print(len(Sentences))
Sentences

# WORD TOKENIZE
Words = word_tokenize(text)
print(len(Words))
Words

# FREEQUENCY DISTRIBUTION
from nltk.probability import FreqDist

fdist = FreqDist(Words)
fdist.most_common(10)

# PLOT THE GRAPH
import matplotlib.pyplot as plt

fdist.plot(10)

# REMOVE PUNCTUATION MARKS
Words_no_punc = []

for w in Words:
    if w.isalpha():
        Words_no_punc.append(w.lower())
        
print(Words_no_punc)

# PLOTTING GRAPH WITHOUT PUNCTUATION MARK
fdist = FreqDist(Words_no_punc)
fdist.most_common(10)
fdist.plot(10)

# LIST OF STOPWORDS
nltk.download('stopwords')
from nltk.corpus import stopwords

stopwords = stopwords.words('english')

# REMOVING STOPWORDS
clean_words = []

for w in Words_no_punc:
    if w not in stopwords:
        clean_words.append(w)

# FINAL FREEQUENCY DISTRIBUTION
fdist = FreqDist(clean_words)
fdist.most_common(15)
fdist.plot(10)

# WORDCLOUD
from wordcloud import WordCloud

wordcloud = WordCloud().generate(text)

plt.figure(figsize = (12, 12))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

# WORDCLOUD OF CIRCLE SHAPE
import numpy as np
from PIL import Image

char = np.array(Image.open('Circle.png'))

wordcloud = WordCloud(background_color="black", mask = char).generate(text)

plt.imshow(wordcloud)
plt.axis("off")
plt.show()

# STEAMMING
from nltk.stem import PorterStemmer

porter = PorterStemmer()
word_list_1 = ["Study", "Studying", "Studies", "Studied"]

for w in word_list_1:
    print(porter.stem(w))

word_list_2 = ["Studies", "Leaves", "Decreases", "Plays"]

for w in word_list_2:
    print(porter.stem(w))

# OTHER STEMMER
from nltk.stem import SnowballStemmer
SnowballStemmer.languages

# LEMMATIZING
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
print(stemmer.stem('studies'))

from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize('studies'))

word_list_3 = ["am", "is", "are", "was", "were"]

# Verb
for w in word_list_3:
    print(lemmatizer.lemmatize(w, pos = "v"))

# Noun
for w in word_list_3:
    print(lemmatizer.lemmatize(w, pos = "n"))

# LEMMATIZER WITH DIFFERENT POS VALUE

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

print(lemmatizer.lemmatize('studying', pos = "v"))
print(lemmatizer.lemmatize('studying', pos = "n"))
print(lemmatizer.lemmatize('studying', pos = "a"))
print(lemmatizer.lemmatize('studying', pos = "r"))

# POS TAGGING
nltk.download('averaged_perceptron_tagger')

tag = nltk.pos_tag(['Studying', 'Study'])
print(tag)

Sentence = 'A very beautiful young lady is walking on the beach'

tokenized_words = word_tokenize(Sentence)

for words in tokenized_words:
    tagged_words = nltk.pos_tag(tokenized_words)

tagged_words

# CHUCKING

# NER : NAMED ENTITY RECOGNITION

Sentence = "Mr. Smith made a deal on a beach of Switzerland near WHO"

tokenized_words = word_tokenize(Sentence)

for w in tokenized_words:
    tagged_words = nltk.pos_tag(tokenized_words)

tagged_words
nltk.download('maxent_ne_chunker')
nltk.download('words')

N_E_R = nltk.ne_chunk(tagged_words, binary = True)
print(N_E_R)
N_E_R.draw()
#  It only shows whether a particular word is named entity or not.

N_E_R = nltk.ne_chunk(tagged_words, binary = False)
print(N_E_R)
N_E_R.draw()
# SHOW TYPE OF NAMED ENTITY

# WORDNET
from nltk.corpus import wordnet

# We can check how many different definitions
# of a word are available in Wordnet.
for words in wordnet.synsets('Fun'):
    print(words)
    
# We can also check the meaning of those different definitions.

for words in wordnet.synsets('Fun'):
    for lemma in words.lemmas():
        print(lemma)
    print("\n")


word = wordnet.synsets("Play")[0]
print(word.name())

# CHECKING DEFINITIONS
print(word.definition())
print(word.examples())


for words in wordnet.synsets('Fun'):
    print(words.name())
    print(words.definition())
    print(words.examples())
    
    for lemma in words.lemmas():
        print(lemma)
    print("\n")

# HYPERNYMS  : GIVES MORE ABSTRACT TERM FOR A WORD

word = wordnet.synsets("Play")[0]
print(word.hypernyms())

# HYPONYMS  : GIVES MORE SPECIFIC TERM FOR A WORD

word = wordnet.synsets("Play")[0]
word.hyponyms()

# GET A NAME ONLY 
print(word.lemmas()[0].name())

# SYNONYMS 
synonyms = []

for words in wordnet.synsets('Fun'):
    for lemma in words.lemmas():
        synonyms.append(lemma.name())
        
synonyms

# ANTONYMS

antonysm = []

for words in wordnet.synsets('Natural'):
    for lemma in words.lemmas():
        if lemma.antonyms():
            antonysm.append(lemma.antonyms()[0].name())
            
antonysm            

# ANTONYMS & SYNONYMS

synonyms = []
antonysm = []

for words in wordnet.synsets('Natural'):
    for lemma in words.lemmas():
        synonyms.append(lemma.name())
        if lemma.antonyms():
            antonysm.append(lemma.antonyms()[0].name())

# FINDING SIMILARITY BETWEEN WORDS

word_1 = wordnet.synsets("ship", "n")[0]
word_2 = wordnet.synsets("boat", "n")[0]
word_3 = wordnet.synsets("bike", "n")[0]

print(word_1.wup_similarity(word_2))
print(word_1.wup_similarity(word_3))

# BAG OF WORDS

from sklearn.feature_extraction.text import CountVectorizer 

sentences = ["Jim and Pam travelled by the bus: ", "The train was late", 
"The flight was full. Travelling by flight is expensive"] 
cv = CountVectorizer() 

# GENERATING BAG OF WORDS
B_O_W = cv.fit_transform(sentences).toarray() 

# Total words with their index in model 
print (cv.vocabulary_) 
print("\n") 
# Features . 
print (cv.get_feature_names()) 
print("\n") 
# Show the output 
print(B_O_W) 

# TF-IDF = TERM FREEQUENCY - INVERSE DOCUMENT FREEQUENCY

from sklearn.feature_extraction.text import TfidfVectorizer 

Sentences = ['This is the first document' , 'This document is the second document']
Sentences 
vectorizer = TfidfVectorizer(norm = None)
# Generating output for TF IDF . 
X = vectorizer.fit_transform(Sentences).toarray() 
# TotaL words with their index in model 
print (vectorizer.vocabulary_) 
print("\n") 
# Features  
print (vectorizer.get_feature_names()) 
print("\n") 
# Show the output 
print(X) 
