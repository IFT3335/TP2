from time import time
from betterer_extract import *
from perceptron import Perceptron
from forest import Forest
from tree import Tree
from naive import Naive
from svms import SvmLinear, SvmPoly, SvmRbf, SvmSig


def test_naive():
    start_time_i: float = time()
    naive: Naive = Naive(xcount2_train, ycount2_train, xcount2_test, ycount2_test)
    naive_score: float = naive.score()
    elapsed_time_i: float = time() - start_time_i
    print("score naive :", naive_score, "- temps :", elapsed_time_i, "s")
    return naive_score, "naive"


def tests_tree():
    score_list = []
    start_time: float = time()
    for i in range(1, 6):
        start_time_i: float = time()
        tree: Tree = Tree(xcount2_train, ycount2_train, xcount2_test, ycount2_test)
        tree_score: float = tree.score()
        elapsed_time_i: float = time() - start_time_i
        score_list.append(tree_score)
        print(f"score tree {i}/5:", tree_score, "- temps :", elapsed_time_i, "s")
    elapsed_time: float = time() - start_time
    print("Temps total tree:", elapsed_time)
    mean: float = float(np.mean(score_list))
    print("Ecart type des scores de tree:", np.std(score_list), "moyenne :", mean)
    return mean, "tree"


def tests_forest():
    score_list = []
    start_time: float = time()
    for i in range(1, 6):
        start_time_i: float = time()
        forest: Forest = Forest(xcount2_train, ycount2_train, xcount2_test, ycount2_test)
        forest_score: float = forest.score()
        elapsed_time_i: float = time() - start_time_i
        score_list.append(forest_score)
        print(f"score forest {i}/5:", forest_score, "- temps :", elapsed_time_i, "s")
    elapsed_time: float = time() - start_time
    print("Temps total forest:", elapsed_time)
    mean: float = float(np.mean(score_list))
    print("Ecart type des scores de forest:", np.std(score_list), "moyenne :", mean)
    return mean, "forest"


def tests_svm():
    score_list = []
    start_time: float = time()
    svm_linear: SvmLinear = SvmLinear(xcount2_train, ycount2_train, xcount2_test, ycount2_test)
    svm_linear_score: float = svm_linear.score()
    elapsed_time: float = time() - start_time
    score_list.append((svm_linear_score, "svm_linear"))
    print("score svm_linear :", svm_linear_score, "- temps :", elapsed_time, "s")

    start_time: float = time()
    svm_poly: SvmPoly = SvmPoly(xcount2_train, ycount2_train, xcount2_test, ycount2_test)
    svm_poly_score: float = svm_poly.score()
    elapsed_time: float = time() - start_time
    score_list.append((svm_poly_score, "svm_poly"))
    print("score svm_poly :", svm_poly_score, "- temps :", elapsed_time, "s")

    start_time: float = time()
    svm_rbf: SvmRbf = SvmRbf(xcount2_train, ycount2_train, xcount2_test, ycount2_test)
    svm_rbf_score: float = svm_rbf.score()
    elapsed_time: float = time() - start_time
    score_list.append((svm_rbf_score, "svm_rbf"))
    print("score svm_rbf :", svm_rbf_score, "- temps :", elapsed_time, "s")

    start_time: float = time()
    svm_sig: SvmSig = SvmSig(xcount2_train, ycount2_train, xcount2_test, ycount2_test)
    svm_sig_score: float = svm_sig.score()
    elapsed_time: float = time() - start_time
    score_list.append((svm_sig_score, "svm_sig"))
    print("score svm_sig :", svm_sig_score, "- temps :", elapsed_time, "s")

    return max(score_list)


def tests_perceptron():
    score_list = []
    start_time: float = time()
    for i in range(1, 6):
        start_time_i: float = time()
        perceptron: Perceptron = Perceptron(xcount2_train, ycount2_train, xcount2_test, ycount2_test)
        perceptron_score: float = perceptron.score()
        elapsed_time_i: float = time() - start_time_i
        score_list.append(perceptron_score)
        print(f"score perceptron {i}/5:", perceptron_score, "- temps :", elapsed_time_i, "s")
    elapsed_time: float = time() - start_time
    print("Temps total perceptron:", elapsed_time)
    mean: float = float(np.mean(score_list))
    print("Ecart type des scores de perceptron:", np.std(score_list), "moyenne :", mean)
    return mean, "perceptron"




