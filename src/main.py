from betterer_extract import *
from naive import Naive
from tree import Tree
from forest import Forest
from svms import SvmLinear, SvmPoly, SvmRbf, SvmSig


def main():
    naive: Naive = Naive(xcount2_train, ycount2_train, xcount2_test, ycount2_test)
    print("score naive : " + str(naive.score()))

    tree: Tree = Tree(xcount2_train, ycount2_train, xcount2_test, ycount2_test)
    print("score tree : " + str(tree.score()))

    forest: Forest = Forest(xcount2_train, ycount2_train, xcount2_test, ycount2_test)
    print("score forest : " + str(forest.score()))

    svm_linear: SvmLinear = SvmLinear(xcount2_train, ycount2_train, xcount2_test, ycount2_test)
    print("score svm_linear : " + str(svm_linear.score()))

    svm_poly: SvmPoly = SvmPoly(xcount2_train, ycount2_train, xcount2_test, ycount2_test)
    print("score svm_poly : " + str(svm_poly.score()))

    svm_rbf: SvmRbf = SvmRbf(xcount2_train, ycount2_train, xcount2_test, ycount2_test)
    print("score svm_rbf : " + str(svm_rbf.score()))

    svm_sig: SvmSig = SvmSig(xcount2_train, ycount2_train, xcount2_test, ycount2_test)
    print("score svm_sig : " + str(svm_sig.score()))


if __name__ == "__main__":
    main()

