from time import time

from betterer_extract import *
from forest import Forest
from naive import Naive
from perceptron import Perceptron
from svms import SvmLinear, SvmPoly, SvmRbf, SvmSig
from tree import Tree


def main():
    total_time_start: float = time()
    score_list = []

    start_time: float = time()
    naive: Naive = Naive(xcount2_train, ycount2_train, xcount2_test, ycount2_test)
    naive_score: float = naive.score()
    elapsed_time: float = time() - start_time
    score_list.append((naive_score, "naive", elapsed_time))
    print("score naive :", naive_score, "- temps :", elapsed_time, "s")

    start_time: float = time()
    tree: Tree = Tree(xcount2_train, ycount2_train, xcount2_test, ycount2_test)
    tree_score: float = tree.score()
    elapsed_time: float = time() - start_time
    score_list.append((tree_score, "tree", elapsed_time))
    print("score tree :", tree_score, "- temps :", elapsed_time, "s")

    start_time: float = time()
    forest: Forest = Forest(xcount2_train, ycount2_train, xcount2_test, ycount2_test)
    forest_score: float = forest.score()
    elapsed_time: float = time() - start_time
    score_list.append((forest_score, "forest", elapsed_time))
    print("score forest :", forest_score, "- temps :", elapsed_time, "s")

    start_time: float = time()
    svm_linear: SvmLinear = SvmLinear(xcount2_train, ycount2_train, xcount2_test, ycount2_test)
    svm_linear_score: float = svm_linear.score()
    elapsed_time: float = time() - start_time
    score_list.append((svm_linear_score, "svm_linear", elapsed_time))
    print("score svm_linear :", svm_linear_score, "- temps :", elapsed_time, "s")

    start_time: float = time()
    svm_poly: SvmPoly = SvmPoly(xcount2_train, ycount2_train, xcount2_test, ycount2_test)
    svm_poly_score: float = svm_poly.score()
    elapsed_time: float = time() - start_time
    score_list.append((svm_poly_score, "svm_poly", elapsed_time))
    print("score svm_poly :", svm_poly_score, "- temps :", elapsed_time, "s")

    start_time: float = time()
    svm_rbf: SvmRbf = SvmRbf(xcount2_train, ycount2_train, xcount2_test, ycount2_test)
    svm_rbf_score: float = svm_rbf.score()
    elapsed_time: float = time() - start_time
    score_list.append((svm_rbf_score, "svm_rbf", elapsed_time))
    print("score svm_rbf :", svm_rbf_score, "- temps :", elapsed_time, "s")

    start_time: float = time()
    svm_sig: SvmSig = SvmSig(xcount2_train, ycount2_train, xcount2_test, ycount2_test)
    svm_sig_score: float = svm_sig.score()
    elapsed_time: float = time() - start_time
    score_list.append((svm_sig_score, "svm_sig", elapsed_time))
    print("score svm_sig :", svm_sig_score, "- temps :", elapsed_time, "s")

    start_time: float = time()
    perceptron: Perceptron = Perceptron(xcount2_train, ycount2_train, xcount2_test, ycount2_test)
    perceptron_score: float = perceptron.score()
    elapsed_time: float = time() - start_time
    score_list.append((perceptron_score, "perceptron", elapsed_time))
    print("score perceptron :", perceptron_score, "- temps :", elapsed_time, "s")

    score_max: float
    algorithme: str
    (score_max, algorithme, temps) = max(score_list)
    print("Meilleur score :", score_max, "avec", algorithme, "en", temps, "s")

    total_time_elapsed: float = time() - total_time_start

    print("Temps total d'exécution :", total_time_elapsed, "s")


if __name__ == "__main__":
    main()
