from sklearn import svm


class SvmLinear:
    def __init__(self, xcount2_train, ycount2_train, xcount2_test, ycount2_test) -> None:
        self.__clf: svm.SVC = svm.SVC(kernel='linear', decision_function_shape='ovo')
        self.__xcount2_train = xcount2_train
        self.__ycount2_train = ycount2_train
        self.__xcount2_test = xcount2_test
        self.__ycount2_test = ycount2_test
        self.__fit()

    def __fit(self) -> None:
        self.__clf.fit(self.__xcount2_train, self.__ycount2_train)

    def score(self) -> float:
        return self.__clf.score(self.__xcount2_test, self.__ycount2_test)


class SvmPoly:
    def __init__(self, xcount2_train, ycount2_train, xcount2_test, ycount2_test) -> None:
        self.__clf: svm.SVC = svm.SVC(kernel='poly', decision_function_shape='ovo')
        self.__xcount2_train = xcount2_train
        self.__ycount2_train = ycount2_train
        self.__xcount2_test = xcount2_test
        self.__ycount2_test = ycount2_test
        self.__fit()

    def __fit(self) -> None:
        self.__clf.fit(self.__xcount2_train, self.__ycount2_train)

    def score(self) -> float:
        return self.__clf.score(self.__xcount2_test, self.__ycount2_test)

    class SvmRbf:
        def __init__(self, xcount2_train, ycount2_train, xcount2_test, ycount2_test) -> None:
            self.__clf: svm.SVC = svm.SVC(kernel='rbf', decision_function_shape='ovo')
            self.__xcount2_train = xcount2_train
            self.__ycount2_train = ycount2_train
            self.__xcount2_test = xcount2_test
            self.__ycount2_test = ycount2_test
            self.__fit()

        def __fit(self) -> None:
            self.__clf.fit(self.__xcount2_train, self.__ycount2_train)

        def score(self) -> float:
            return self.__clf.score(self.__xcount2_test, self.__ycount2_test)


class SvmRbf:
    def __init__(self, xcount2_train, ycount2_train, xcount2_test, ycount2_test) -> None:
        self.__clf: svm.SVC = svm.SVC(kernel='rbf', decision_function_shape='ovo')
        self.__xcount2_train = xcount2_train
        self.__ycount2_train = ycount2_train
        self.__xcount2_test = xcount2_test
        self.__ycount2_test = ycount2_test
        self.__fit()

    def __fit(self) -> None:
        self.__clf.fit(self.__xcount2_train, self.__ycount2_train)

    def score(self) -> float:
        return self.__clf.score(self.__xcount2_test, self.__ycount2_test)


class SvmSig:
    def __init__(self, xcount2_train, ycount2_train, xcount2_test, ycount2_test) -> None:
        self.__clf: svm.SVC = svm.SVC(kernel='sigmoid', decision_function_shape='ovo')
        self.__xcount2_train = xcount2_train
        self.__ycount2_train = ycount2_train
        self.__xcount2_test = xcount2_test
        self.__ycount2_test = ycount2_test
        self.__fit()

    def __fit(self) -> None:
        self.__clf.fit(self.__xcount2_train, self.__ycount2_train)

    def score(self) -> float:
        return self.__clf.score(self.__xcount2_test, self.__ycount2_test)
