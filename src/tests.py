from time import time
from betterer_extract import *
from perceptron import Perceptron
from forest import Forest
from tree import Tree
from naive import Naive
from svms import SvmLinear, SvmPoly, SvmRbf, SvmSig


def test_naive():
    print("--- Tests Naive ---")
    start_time_i: float = time()
    naive: Naive = Naive(xcount2_train, ycount2_train, xcount2_test, ycount2_test)
    naive_score: float = naive.score()
    elapsed_time_i: float = time() - start_time_i
    print("Score naive :", naive_score, "- temps :", f"{elapsed_time_i}s")
    print("\n")
    return naive_score, "naive"


def tests_tree():
    print("--- Tests Tree ---")
    score_list = []
    start_time: float = time()
    for i in range(1, 6):
        start_time_i: float = time()
        tree: Tree = Tree(xcount2_train, ycount2_train, xcount2_test, ycount2_test)
        tree_score: float = tree.score()
        elapsed_time_i: float = time() - start_time_i
        score_list.append(tree_score)
        print(f"Score tree {i}/5:", tree_score, "- temps :", f"{elapsed_time_i}s")
    elapsed_time: float = time() - start_time
    mean: float = float(np.mean(score_list))
    print("Moyenne :", mean, "- Ecart type :", np.std(score_list))
    print("Temps total tree:", f"{elapsed_time}s")
    print("\n")
    return mean, "tree"


def tests_forest():
    print("--- Tests Forest ---")
    score_list = []
    start_time: float = time()
    for i in range(1, 6):
        start_time_i: float = time()
        forest: Forest = Forest(xcount2_train, ycount2_train, xcount2_test, ycount2_test)
        forest_score: float = forest.score()
        elapsed_time_i: float = time() - start_time_i
        score_list.append(forest_score)
        print(f"Score forest {i}/5:", forest_score, "- temps :", f"{elapsed_time_i}s")
    elapsed_time: float = time() - start_time
    mean: float = float(np.mean(score_list))
    print("Moyenne :", mean, "- Ecart type :", np.std(score_list))
    print("Temps total forest:", f"{elapsed_time}s")
    print("\n")
    return mean, "forest"


def tests_svm():
    print("--- Tests Svm ---")
    start_time_t: float = time()
    score_list = []
    start_time: float = time()
    svm_linear: SvmLinear = SvmLinear(xcount2_train, ycount2_train, xcount2_test, ycount2_test)
    svm_linear_score: float = svm_linear.score()
    elapsed_time: float = time() - start_time
    score_list.append((svm_linear_score, "svm_linear"))
    print("Score svm_linear :", svm_linear_score, "- temps :", f"{elapsed_time}s")

    start_time: float = time()
    svm_poly: SvmPoly = SvmPoly(xcount2_train, ycount2_train, xcount2_test, ycount2_test)
    svm_poly_score: float = svm_poly.score()
    elapsed_time: float = time() - start_time
    score_list.append((svm_poly_score, "svm_poly"))
    print("Score svm_poly :", svm_poly_score, "- temps :", f"{elapsed_time}s")

    start_time: float = time()
    svm_rbf: SvmRbf = SvmRbf(xcount2_train, ycount2_train, xcount2_test, ycount2_test)
    svm_rbf_score: float = svm_rbf.score()
    elapsed_time: float = time() - start_time
    score_list.append((svm_rbf_score, "svm_rbf"))
    print("Score svm_rbf :", svm_rbf_score, "- temps :", f"{elapsed_time}s")

    start_time: float = time()
    svm_sig: SvmSig = SvmSig(xcount2_train, ycount2_train, xcount2_test, ycount2_test)
    svm_sig_score: float = svm_sig.score()
    elapsed_time: float = time() - start_time
    score_list.append((svm_sig_score, "svm_sig"))
    print("Score svm_sig :", svm_sig_score, "- temps :", f"{elapsed_time}s")
    score_max, algo = max(score_list)

    elapsed_time_t: float = time() - start_time_t

    print("Svm max :", algo, "- score :", score_max)
    print("Temps total Svms:", f"{elapsed_time_t}s")
    print("\n")
    return score_max, algo


def tests_perceptron():
    print("--- Tests Perceptron ---")
    score_list = []
    start_time: float = time()
    for i in range(1, 6):
        start_time_i: float = time()
        perceptron: Perceptron = Perceptron(xcount2_train, ycount2_train, xcount2_test, ycount2_test)
        perceptron_score: float = perceptron.score()
        elapsed_time_i: float = time() - start_time_i
        score_list.append(perceptron_score)
        print(f"Score perceptron {i}/5:", perceptron_score, "- temps :", f"{elapsed_time_i}s")
    elapsed_time: float = time() - start_time
    mean: float = float(np.mean(score_list))
    print("Moyenne :", mean, "- Ecart type :", np.std(score_list))
    print("Temps total perceptron:", f"{elapsed_time}s")
    print("\n")
    return mean, "perceptron"




