from lightgbm import LGBMRegressor, early_stopping

from backbone.train_utils.trainable import Trainable
from backbone.train_utils.trainable_factory import TrainableFactory
from backbone.train_utils.trainable.optuna import MetabolomicsPredictionObjective, OptunaTrainable

from backbone.lgbm.train import mse_wo_nans


class OptunaLGBMFactory(TrainableFactory):
    def __init__(self, conf, model_class=None):
        self.conf = conf
        scoring_func = conf['train']['trainable']['args']['scoring']
        self.scoring_func = scoring_func if scoring_func is not None else mse_wo_nans
        self.model_class = model_class if model_class is not None else LGBMRegressor

        self.additional_fit_params = {
            "eval_metric": self.scoring_func
        }

    def create(self) -> Trainable:
        objective = MetabolomicsPredictionObjective(self.scoring_func, self.model_class, self.conf)
        trainable = OptunaTrainable(objective, self.conf, **self.additional_fit_params)

        return trainable
