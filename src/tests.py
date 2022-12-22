from time import time

import numpy as np

from forest import Forest
from naive import Naive
from perceptron import Perceptron
from svm import Svm
from tree import Tree


def test_naive(xcount2_train, ycount2_train, xcount2_test, ycount2_test, **kwargs):
    print(f"--- Tests Naive - paramètres : {kwargs} ---")
    start_time_i: float = time()
    naive: Naive = Naive(xcount2_train, ycount2_train, xcount2_test, ycount2_test, **kwargs)
    naive_score: float = naive.score()
    elapsed_time_i: float = time() - start_time_i
    print(f"Score naive: {naive_score:.3f} - temps : {elapsed_time_i:.3f}s\n")
    return naive_score, "naive", kwargs


def tests_tree(xcount2_train, ycount2_train, xcount2_test, ycount2_test, **kwargs):
    print(f"--- Tests Tree - paramètres : {kwargs} ---")
    score_list = []
    time_list = []
    start_time: float = time()
    for i in range(1, 6):
        start_time_i: float = time()
        tree: Tree = Tree(xcount2_train, ycount2_train, xcount2_test, ycount2_test, **kwargs)
        tree_score: float = tree.score()
        elapsed_time_i: float = time() - start_time_i
        score_list.append(tree_score)
        time_list.append(elapsed_time_i)
        print(f"Score tree {i}/5: {tree_score:.3f} - temps : {elapsed_time_i:.3f}s")
    elapsed_time: float = time() - start_time
    mean: float = float(np.mean(score_list))
    print(f"Moyenne des scores : {mean:.3f} - Ecart type : {np.std(score_list):.3f}")
    print(f"Temps moyen : {np.mean(time_list):.3f}s - Ecart Type : {np.std(time_list):.3f}")
    print("Temps total tree:", f"{elapsed_time:.3f}s\n")
    return mean, "tree", kwargs


def tests_forest(xcount2_train, ycount2_train, xcount2_test, ycount2_test, **kwargs):
    print(f"--- Tests Forest - paramètres : {kwargs} ---")
    score_list = []
    time_list = []
    start_time: float = time()
    for i in range(1, 6):
        start_time_i: float = time()
        forest: Forest = Forest(xcount2_train, ycount2_train, xcount2_test, ycount2_test, **kwargs)
        forest_score: float = forest.score()
        elapsed_time_i: float = time() - start_time_i
        score_list.append(forest_score)
        time_list.append(elapsed_time_i)
        print(f"Score forest {i}/5: {forest_score:.3f} - temps : {elapsed_time_i:.3f}s")
    elapsed_time: float = time() - start_time
    mean: float = float(np.mean(score_list))
    print(f"Moyenne des scores : {mean:.3f} - Ecart type : {np.std(score_list):.3f}")
    print(f"Temps moyen : {np.mean(time_list):.3f}s - Ecart Type : {np.std(time_list):.3f}")
    print("Temps total forest:", f"{elapsed_time:.3f}s\n")
    return mean, "forest", kwargs


def tests_svm(xcount2_train, ycount2_train, xcount2_test, ycount2_test, **kwargs):
    print(f"--- Tests Svm - paramètres : {kwargs} ---")
    start_time_i: float = time()
    svm: Svm = Svm(xcount2_train, ycount2_train, xcount2_test, ycount2_test, **kwargs)
    svm_score: float = svm.score()
    elapsed_time_i: float = time() - start_time_i
    print(f"Score svm: {svm_score:.3f} - temps : {elapsed_time_i:.3f}s\n")
    return svm_score, "svm", kwargs


def tests_perceptron(xcount2_train, ycount2_train, xcount2_test, ycount2_test, **kwargs):
    print(f"--- Tests Perceptron - paramètres : {kwargs} ---")
    score_list = []
    time_list = []
    start_time: float = time()
    for i in range(1, 6):
        start_time_i: float = time()
        perceptron: Perceptron = Perceptron(xcount2_train, ycount2_train, xcount2_test, ycount2_test, **kwargs)
        perceptron_score: float = perceptron.score()
        elapsed_time_i: float = time() - start_time_i
        score_list.append(perceptron_score)
        time_list.append(elapsed_time_i)
        print(f"Score perceptron {i}/5: {perceptron_score:.3f} - temps : {elapsed_time_i:.3f}s")
    elapsed_time: float = time() - start_time
    mean: float = float(np.mean(score_list))
    print(f"Moyenne des scores : {mean:.3f} - Ecart type : {np.std(score_list):.3f}")
    print(f"Temps moyen : {np.mean(time_list):.3f}s - Ecart Type : {np.std(time_list):.3f}")
    print("Temps total perceptron:", f"{elapsed_time:.3f}s\n")
    return mean, "perceptron", kwargs
