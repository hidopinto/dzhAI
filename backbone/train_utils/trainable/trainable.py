from abc import abstractmethod


class Trainable:
    @abstractmethod
    def fit(self, x, y=None, **fit_params):
        pass

    @abstractmethod
    def predict(self, x, **predict_params):
        pass

    @abstractmethod
    def score(self, x, y=None, **score_params):
        pass
