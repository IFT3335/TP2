import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split


raw_corpus = open("interest.acl94.txt").read()
raw_corpus = raw_corpus.replace("[", "").replace("]", "").replace("=", "").split("$$")

raw_phrases = list(map(str.strip, raw_corpus))
list_phrases = list(filter(None, map(str.split, raw_corpus)))


def parser(phrase, target):
    mots = []
    cats = []
    l = None

    for mot_cat in phrase:
        t = mot_cat.split("/")
        if len(t) != 2:
            return None

        mot, cat = t

        if re.match(f"{target}(|s)_[0-9]", mot):
            mot, l = mot.split("_")

        mots.append(mot)
        cats.append(cat)

    return mots, cats, l


word_list_phrases, cat_phrases, labels = list(
    zip(*filter(None, map(lambda phrase: parser(phrase, "interest"), list_phrases)))
)
word_str_phrases = [" ".join(phrase) for phrase in word_list_phrases]
labels = list(map(int, labels))

unique_cats = set()
for cat_phrase_set in map(set, cat_phrases):
    unique_cats |= cat_phrase_set

cat_ids = dict(zip(unique_cats, range(1, len(unique_cats) + 1)))
cat_ids[None] = 0

# legende
#
# raw_phrases       : ["mot1/cat mot2/cat", ...]
# list_phrases      : [["mot1/cat", "mot2/cat"], ...]
# word_list_phrases : [["mot1", "mot2"], ...]
# word_str_phrases  : ["mot1 mot2", ...]
# cat_phrases       : [["cat", "cat"], ...]
# labels            : [1, ...]

count_vectorizer = CountVectorizer()
count_vectorizer.fit_transform(word_str_phrases)

tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit_transform(word_str_phrases)

# a partir d'ici je prepare X et y

def context(word_list_phrase, cat_phrase, regex, fenetre):
    for i, word in enumerate(word_list_phrase):
        if re.match(regex, word):
            break

    word_pre = word_list_phrase[i - fenetre : i]
    word_post = word_list_phrase[i + 1 : i + 1 + fenetre]

    cat_pre = cat_phrase[i - fenetre : i]
    cat_post = cat_phrase[i + 1 : i + 1 + fenetre]

    while len(cat_pre) < fenetre:
        cat_pre = [None] + cat_pre

    while len(cat_post) < fenetre:
        cat_post += [None]

    return word_pre + word_post, cat_pre + cat_post


# extraction des mots et de leur categorie dans une fenetre
def X_creator(word_list_phrases, cat_phrases, regex, fenetre, vectorizer):
    context_phrases, context_cats = tuple(
        zip(*map(lambda words_cats: context(*words_cats, regex, fenetre), zip(word_list_phrases, cat_phrases)))
    )

    context_str_phrases = [" ".join(context_phrase) for context_phrase in context_phrases]
    X_words = vectorizer.transform(context_str_phrases).toarray()

    context_cat_ids = [list(map(lambda c: cat_ids[c], context_cat)) for context_cat in context_cats]
    X_cats = np.array(context_cat_ids)

    return np.hstack((X_words, X_cats))

# si tu veux t'amuser avec différents ensembles de donnees, c'est ici
Xcount2 = X_creator(word_list_phrases, cat_phrases, "interest(|s)", 2, count_vectorizer)
Xtfidf2 = X_creator(word_list_phrases, cat_phrases, "interest(|s)", 5, tfidf_vectorizer)
y = np.array(labels)

Xcount2_train, Xcount2_test, ycount2_train, ycount2_test = train_test_split(Xcount2, y, test_size=0.2, random_state=42)
Xtfidf2_train, Xtfidf2_test, ytfidf2_train, ytfidf2_test = train_test_split(Xtfidf2, y, test_size=0.2, random_state=42)
