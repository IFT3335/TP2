from time import time
from tests import test_naive, tests_tree, tests_forest, tests_svm, tests_perceptron
from betterer_extract import *


def main():
    total_time_all_windows_start: float = time()
    y = np.array(labels)
    max_list = []
    print("----- Tests avec taille fenêtre = 1 -----")
    total_time_start: float = time()
    xcount2 = x_creator(word_list_phrases, cat_phrases, "interest(|s)", 1, count_vectorizer)
    #xtfidf2 = x_creator(word_list_phrases, cat_phrases, "interest(|s)", 5, tfidf_vectorizer)

    xcount2_train, xcount2_test, ycount2_train, ycount2_test = train_test_split(xcount2, y, test_size=0.2, random_state=42)
    #xtfidf2_train, xtfidf2_test, ytfidf2_train, ytfidf2_test = train_test_split(xtfidf2, y, test_size=0.2,random_state=42)


    score_list = (
        test_naive(xcount2_train, ycount2_train, xcount2_test, ycount2_test),
        tests_tree(xcount2_train, ycount2_train, xcount2_test, ycount2_test),
        tests_forest(xcount2_train, ycount2_train, xcount2_test, ycount2_test),
        tests_svm(xcount2_train, ycount2_train, xcount2_test, ycount2_test),
        tests_perceptron(xcount2_train, ycount2_train, xcount2_test, ycount2_test, learning_rate="adaptive", learning_rate_init=0.025, hidden_layer_sizes=(175,))
    )
    score, algo = max(score_list)
    max_list.append((score, algo, "1"))
    print("Meilleur score :", score, "avec", algo)
    total_time_elapsed: float = time() - total_time_start
    print("Temps total d'exécution fenêtre = 1 :", total_time_elapsed, "s")

    print("\n\n----- Tests avec taille fenêtre = 2 -----")
    total_time_start: float = time()
    xcount2 = x_creator(word_list_phrases, cat_phrases, "interest(|s)", 2, count_vectorizer)
    #xtfidf2 = x_creator(word_list_phrases, cat_phrases, "interest(|s)", 5, tfidf_vectorizer)

    xcount2_train, xcount2_test, ycount2_train, ycount2_test = train_test_split(xcount2, y, test_size=0.2, random_state=42)
    #xtfidf2_train, xtfidf2_test, ytfidf2_train, ytfidf2_test = train_test_split(xtfidf2, y, test_size=0.2,random_state=42)

    score_list = (
        test_naive(xcount2_train, ycount2_train, xcount2_test, ycount2_test),
        tests_tree(xcount2_train, ycount2_train, xcount2_test, ycount2_test),
        tests_forest(xcount2_train, ycount2_train, xcount2_test, ycount2_test),
        tests_svm(xcount2_train, ycount2_train, xcount2_test, ycount2_test),
        tests_perceptron(xcount2_train, ycount2_train, xcount2_test, ycount2_test, learning_rate="adaptive", learning_rate_init=0.025, hidden_layer_sizes=(175,))
    )
    score, algo = max(score_list)
    max_list.append((score, algo, "2"))
    print("Meilleur score :", score, "avec", algo)
    total_time_elapsed: float = time() - total_time_start
    print("Temps total d'exécution fenêtre = 2 :", total_time_elapsed, "s")

    print("\n\n----- Tests avec taille fenêtre = 5 -----")
    total_time_start: float = time()
    xcount2 = x_creator(word_list_phrases, cat_phrases, "interest(|s)", 5, count_vectorizer)
    #xtfidf2 = x_creator(word_list_phrases, cat_phrases, "interest(|s)", 5, tfidf_vectorizer)

    xcount2_train, xcount2_test, ycount2_train, ycount2_test = train_test_split(xcount2, y, test_size=0.2, random_state=42)
    #xtfidf2_train, xtfidf2_test, ytfidf2_train, ytfidf2_test = train_test_split(xtfidf2, y, test_size=0.2,random_state=42)

    score_list = (
        test_naive(xcount2_train, ycount2_train, xcount2_test, ycount2_test),
        tests_tree(xcount2_train, ycount2_train, xcount2_test, ycount2_test),
        tests_forest(xcount2_train, ycount2_train, xcount2_test, ycount2_test),
        tests_svm(xcount2_train, ycount2_train, xcount2_test, ycount2_test),
        tests_perceptron(xcount2_train, ycount2_train, xcount2_test, ycount2_test, learning_rate="adaptive", learning_rate_init=0.025, hidden_layer_sizes=(175,))
    )
    score, algo = max(score_list)
    max_list.append((score, algo, "5"))
    print("Meilleur score :", score, "avec", algo)
    total_time_elapsed: float = time() - total_time_start
    print("Temps total d'exécution fenêtre = 5 :", total_time_elapsed, "s")

    score, algo, fenetre = max(max_list)
    print("\n\n----- Bilan -----")
    print("Meilleur score :", score, "avec", algo, "pour la fenêtre de taille", fenetre)
    total_time_elapsed: float = time() - total_time_all_windows_start
    print("Temps total d'exécution :", total_time_elapsed, "s")


if __name__ == "__main__":
    main()
