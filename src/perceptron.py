from sklearn.neural_network import MLPClassifier


class Perceptron:

    def __init__(self, xcount2_train, ycount2_train, xcount2_test, ycount2_test) -> None:
        self.__clf: MLPClassifier = MLPClassifier()
        self.__xcount2_train = xcount2_train
        self.__ycount2_train = ycount2_train
        self.__xcount2_test = xcount2_test
        self.__ycount2_test = ycount2_test
        self.__fit()

    def __fit(self) -> None:
        self.__clf.fit(self.__xcount2_train, self.__ycount2_train)

    def score(self) -> float:
        return self.__clf.score(self.__xcount2_test, self.__ycount2_test)
