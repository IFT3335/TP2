from time import time

from sklearn.model_selection import train_test_split

from betterer_extract import *
from tests import test_naive, tests_tree, tests_forest, tests_svm, tests_perceptron


def main():
    print("----- Tests avec jeu de paramètres 1 -----\n\n")
    total_time_all_windows_start: float = time()
    y = np.array(labels)
    max_list = []
    print("----- Tests avec taille fenêtre = 1 -----\n")
    total_time_start: float = time()
    xcount2 = x_creator(word_list_phrases, cat_phrases, "interest(|s)", 1, count_vectorizer)
    # xtfidf2 = x_creator(word_list_phrases, cat_phrases, "interest(|s)", 5, tfidf_vectorizer)

    xcount2_train, xcount2_test, ycount2_train, ycount2_test = train_test_split(xcount2, y, test_size=0.2,
                                                                                random_state=42)
    # xtfidf2_train, xtfidf2_test, ytfidf2_train, ytfidf2_test = train_test_split(xtfidf2, y, test_size=0.2,
    # random_state=42)

    score_list = (
        test_naive(xcount2_train, ycount2_train, xcount2_test, ycount2_test),
        tests_tree(xcount2_train, ycount2_train, xcount2_test, ycount2_test),
        tests_forest(xcount2_train, ycount2_train, xcount2_test, ycount2_test),
        tests_svm(xcount2_train, ycount2_train, xcount2_test, ycount2_test, kernel='linear',
                  decision_function_shape='ovo'),
        tests_perceptron(xcount2_train, ycount2_train, xcount2_test, ycount2_test, learning_rate="adaptive",
                         learning_rate_init=0.025, hidden_layer_sizes=(175,))
    )
    score, algo, kwargs = max(score_list)
    max_list.append((score, algo, kwargs, "1"))
    print(f"Meilleur score :{score:.3f} avec {algo} et paramètres : {kwargs}")
    total_time_elapsed: float = time() - total_time_start
    print(f"Temps total d'exécution fenêtre = 1 : {total_time_elapsed:.3f}s")

    print("\n\n----- Tests avec taille fenêtre = 2 -----")
    total_time_start: float = time()
    xcount2 = x_creator(word_list_phrases, cat_phrases, "interest(|s)", 2, count_vectorizer)
    # xtfidf2 = x_creator(word_list_phrases, cat_phrases, "interest(|s)", 5, tfidf_vectorizer)

    xcount2_train, xcount2_test, ycount2_train, ycount2_test = train_test_split(xcount2, y, test_size=0.2,
                                                                                random_state=42)
    # xtfidf2_train, xtfidf2_test, ytfidf2_train, ytfidf2_test = train_test_split(xtfidf2, y, test_size=0.2,
    # random_state=42)

    score_list = (
        test_naive(xcount2_train, ycount2_train, xcount2_test, ycount2_test),
        tests_tree(xcount2_train, ycount2_train, xcount2_test, ycount2_test),
        tests_forest(xcount2_train, ycount2_train, xcount2_test, ycount2_test),
        tests_svm(xcount2_train, ycount2_train, xcount2_test, ycount2_test, kernel='linear',
                  decision_function_shape='ovo'),
        tests_perceptron(xcount2_train, ycount2_train, xcount2_test, ycount2_test, learning_rate="adaptive",
                         learning_rate_init=0.025, hidden_layer_sizes=(175,))
    )
    score, algo, kwargs = max(score_list)
    max_list.append((score, algo, kwargs, "2"))
    print(f"Meilleur score :{score:.3f} avec {algo} et paramètres : {kwargs}")
    total_time_elapsed: float = time() - total_time_start
    print(f"Temps total d'exécution fenêtre = 2 : {total_time_elapsed:.3f}s")

    print("\n\n----- Tests avec taille fenêtre = 5 -----")
    total_time_start: float = time()
    xcount2 = x_creator(word_list_phrases, cat_phrases, "interest(|s)", 5, count_vectorizer)
    # xtfidf2 = x_creator(word_list_phrases, cat_phrases, "interest(|s)", 5, tfidf_vectorizer)

    xcount2_train, xcount2_test, ycount2_train, ycount2_test = train_test_split(xcount2, y, test_size=0.2,
                                                                                random_state=42)
    # xtfidf2_train, xtfidf2_test, ytfidf2_train, ytfidf2_test = train_test_split(xtfidf2, y, test_size=0.2,
    # random_state=42)

    score_list = (
        test_naive(xcount2_train, ycount2_train, xcount2_test, ycount2_test),
        tests_tree(xcount2_train, ycount2_train, xcount2_test, ycount2_test),
        tests_forest(xcount2_train, ycount2_train, xcount2_test, ycount2_test),
        tests_svm(xcount2_train, ycount2_train, xcount2_test, ycount2_test, kernel='linear',
                  decision_function_shape='ovo'),
        tests_perceptron(xcount2_train, ycount2_train, xcount2_test, ycount2_test, learning_rate="adaptive",
                         learning_rate_init=0.025, hidden_layer_sizes=(175,))
    )
    score, algo, kwargs = max(score_list)
    max_list.append((score, algo, kwargs, "5"))
    print(f"Meilleur score :{score:.3f} avec {algo} et paramètres : {kwargs}")
    total_time_elapsed: float = time() - total_time_start
    print(f"Temps total d'exécution fenêtre = 5 : {total_time_elapsed:.3f}s")

    score, algo, kwargs, fenetre = max(max_list)
    print("\n\n----- Bilan test des tailles de fenêtres -----")
    print(f"Meilleur score :{score:.3f} avec {algo} et paramètres {kwargs} pour la fenêtre de taille {fenetre}")
    total_time_elapsed: float = time() - total_time_all_windows_start
    print(f"Temps total d'exécution : {total_time_elapsed:.3f}s")

    print("----- Tests avec différents jeux de paramètres -----")
    print("\n\n----- Tests avec taille fenêtre = 2 -----")
    total_time_start: float = time()
    xcount2 = x_creator(word_list_phrases, cat_phrases, "interest(|s)", 2, count_vectorizer)
    # xtfidf2 = x_creator(word_list_phrases, cat_phrases, "interest(|s)", 5, tfidf_vectorizer)

    xcount2_train, xcount2_test, ycount2_train, ycount2_test = train_test_split(xcount2, y, test_size=0.2,
                                                                                random_state=42)
    # xtfidf2_train, xtfidf2_test, ytfidf2_train, ytfidf2_test = train_test_split(xtfidf2, y, test_size=0.2,
    # random_state=42)

    score_list = (
        test_naive(xcount2_train, ycount2_train, xcount2_test, ycount2_test),
        tests_tree(xcount2_train, ycount2_train, xcount2_test, ycount2_test, max_depth=40),
        tests_tree(xcount2_train, ycount2_train, xcount2_test, ycount2_test, max_depth=20),
        tests_tree(xcount2_train, ycount2_train, xcount2_test, ycount2_test, max_depth=5),
        tests_forest(xcount2_train, ycount2_train, xcount2_test, ycount2_test, n_estimators=10),
        tests_forest(xcount2_train, ycount2_train, xcount2_test, ycount2_test, n_estimators=100),
        tests_forest(xcount2_train, ycount2_train, xcount2_test, ycount2_test, n_estimators=1000),
        tests_svm(xcount2_train, ycount2_train, xcount2_test, ycount2_test, kernel='linear',
                  decision_function_shape='ovo'),
        tests_svm(xcount2_train, ycount2_train, xcount2_test, ycount2_test, kernel='poly',
                  decision_function_shape='ovo'),
        tests_svm(xcount2_train, ycount2_train, xcount2_test, ycount2_test, kernel='rbf',
                  decision_function_shape='ovo'),
        tests_svm(xcount2_train, ycount2_train, xcount2_test, ycount2_test, kernel='sigmoid',
                  decision_function_shape='ovo'),
        tests_perceptron(xcount2_train, ycount2_train, xcount2_test, ycount2_test, learning_rate="adaptive",
                         learning_rate_init=0.1, hidden_layer_sizes=(125,)),
        tests_perceptron(xcount2_train, ycount2_train, xcount2_test, ycount2_test, learning_rate="adaptive",
                         learning_rate_init=0.025, hidden_layer_sizes=(125,)),
        tests_perceptron(xcount2_train, ycount2_train, xcount2_test, ycount2_test, learning_rate="adaptive",
                         learning_rate_init=0.025, hidden_layer_sizes=(175,)),
        tests_perceptron(xcount2_train, ycount2_train, xcount2_test, ycount2_test, learning_rate="adaptive",
                         learning_rate_init=0.025, hidden_layer_sizes=(225,))
    )
    score, algo, kwargs = max(score_list)
    print(f"Meilleur score :{score:.3f} avec {algo} et paramètres {kwargs}")
    total_time_elapsed: float = time() - total_time_start
    print(f"Temps total d'exécution fenêtre = 2 : {total_time_elapsed:.3f}s")

    total_time_elapsed: float = time() - total_time_all_windows_start
    print(f"Temps total d'exécution du programme: {total_time_elapsed:.3f}s")


if __name__ == "__main__":
    main()
