from sklearn.model_selection import GridSearchCV

from backbone.train_utils.trainable.trainable import Trainable


class GridSearchTrainable(Trainable):
    def __init__(self, model_blueprint, *args, **kwargs):
        self.searcher = GridSearchCV(estimator=model_blueprint, *args, **kwargs)

    def fit(self, x, y=None, **fit_params):
        return self.searcher.fit(x, y, **fit_params)

    def predict(self, x, **predict_params):
        return self.searcher.predict(x, **predict_params)

    def score(self, x, y=None, **score_params):
        return self.searcher.score(x, y)
