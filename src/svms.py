from sklearn import svm


class Svm:
    def __init__(self, x_train, y_train, x_test, y_test, **kwargs) -> None:
        self.__clf: svm.SVC = svm.SVC(**kwargs)
        self.__x_train = x_train
        self.__y_train = y_train
        self.__x_test = x_test
        self.__y_test = y_test
        self.__fit()

    def __fit(self) -> None:
        self.__clf.fit(self.__x_train, self.__y_train)

    def score(self) -> float:
        return self.__clf.score(self.__x_test, self.__y_test)
