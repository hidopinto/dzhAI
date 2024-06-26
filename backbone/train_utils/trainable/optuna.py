import os

from optuna import create_study
from lightgbm import Dataset

from backbone.train_utils.trainable.trainable import Trainable


class MetabolomicsPredictionObjective:
    def __init__(self, score_func, model_class, conf):
        self.score_func = score_func
        self.conf = conf

        self.train_data = None
        self.train_labels = None

        self.val_data = None
        self.val_label = None
        self.eval_set = None

        self.fit_args = []
        self.fit_kwargs = {}

        self.model_class = model_class
        self.model = None

    def __call__(self, trial):
        optuna_params = {
            # regular params
            'objective': 'regression',
            'metric': 'custom',
            'verbosity': -1,
            'verbose': -1,
            'boosting_type': 'gbdt',
            # tuner params
            'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
            'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
            'num_leaves': trial.suggest_int('num_leaves', 2, 1024),  # 256
            'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            # structure params
            'max_depth': trial.suggest_int('max_depth', -1, 15),
            'learning_rate': trial.suggest_uniform('learning_rate', 1e-6, 0.1),
            'n_estimator': trial.suggest_int('n_estimator', 20, 5000)
        }

        self.model = self.model_class(**optuna_params)

        self.model.fit(
            self.train_data,
            self.train_labels,
            eval_set=self.eval_set,
            *self.fit_args,
            **self.fit_kwargs
        )

        if self.val_data is not None:
            score_data = self.val_data
            score_label = self.val_label
        else:
            score_data = self.train_data
            score_label = self.train_labels
        preds = self.model.predict(score_data)
        model_score = self.score_func(score_label, preds)[1]

        return model_score

    def set_fit_args(self, *fit_args, **fit_kwargs):
        self.fit_args = fit_args
        self.fit_kwargs = fit_kwargs

    def set_train_data(self, x, y):
        self.train_data = x
        self.train_labels = y

    def set_val_data(self, x, y):
        self.val_data = x
        self.val_label = y

        self.eval_set = [(self.val_data, self.val_label)]

    def get_model(self):
        if self.model is None:
            raise ValueError("self.model is None")

        return self.model


class OptunaTrainable(Trainable):
    def __init__(self, objective, conf, **additional_fit_params):
        self.study = create_study(direction=conf['lgbm']['train']['direction'])
        self.objective = objective

        self.trained_model = None
        self.optimized = False

        self.additional_fit_params = additional_fit_params

        self.scoring_func = conf['train']['trainable']['args']['scoring']

        self.conf = conf

    def optimize(self, x, y, val_x=None, val_y=None, **fit_params):
        self.objective.set_train_data(x, y)
        if val_y is not None:
            self.objective.set_val_data(val_x, val_y)
        self.objective.set_fit_args(**fit_params)

        num_cores = self.conf['lgbm']['train']['num_cores'] if 'num_cores' in self.conf['lgbm']['train'] else -1
        if num_cores > 0:
            os.environ['OMP_NUM_THREADS'] = f"{num_cores}"

        self.study.optimize(self.objective, n_trials=self.conf['lgbm']['train']['n_trials'])
        self.optimized = True

    def fit(self, x, y=None, val_x=None, val_y=None, **fit_params):
        if not self.optimized:
            self.optimize(x, y, val_x, val_y, **fit_params, **self.additional_fit_params)

        self.trained_model = self.objective.get_model()

        # return self.trained_model
        return self

    def predict(self, x, **predict_params):
        return self.trained_model.predict(x, **predict_params)

    def score(self, x, y=None, **score_params):
        return self.scoring_func(y, self.trained_model.predict(x), **score_params)[1]
