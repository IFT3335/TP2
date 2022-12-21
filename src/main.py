from time import time
from tests import test_naive, tests_tree, tests_forest, tests_svm, tests_perceptron


def main():
    total_time_start: float = time()
    score_list = [test_naive(), tests_tree(), tests_forest(), tests_svm(), tests_perceptron()]
    score, algo = max(score_list)
    print("Meilleur score :", score, "avec", algo)
    total_time_elapsed: float = time() - total_time_start
    print("Temps total d'exécution :", total_time_elapsed, "s")


if __name__ == "__main__":
    main()
