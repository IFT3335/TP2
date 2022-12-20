from sklearn import svm


class SvmPoly:
    def __init__(self, xcount2_train, ycount2_train, xcount2_test, ycount2_test):
        self.__clf: svm.SVC = svm.SVC(kernel='poly', decision_function_shape='ovo')
        self.__xcount2_train = xcount2_train
        self.__ycount2_train = ycount2_train
        self.__xcount2_test = xcount2_test
        self.__ycount2_test = ycount2_test
        self.__fit()

    def __fit(self):
        self.__clf.fit(self.__xcount2_train, self.__ycount2_train)

    def score(self):
        return self.__clf.score(self.__xcount2_test, self.__ycount2_test)
