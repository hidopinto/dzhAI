import traceback
from abc import ABC, abstractmethod

from copy import deepcopy
from sklearn.model_selection import cross_validate
from tqdm import tqdm, tqdm_notebook

from backbone.train_utils.trainable import Trainable


class Trainer(ABC):
    @abstractmethod
    def train(self, x, y):
        pass


class SingleColumnTrainer(Trainer):
    def __init__(self, model, conf):
        self.model = model
        self.conf = conf

        self.run_cross_validation = self.conf['train']['run_cross_validation']
        self.cv_params = conf['train']['cv_params'].copy()

    def train(self, x, y, val_x=None, val_y=None):
        training_results = None

        if self.run_cross_validation:
            training_results = cross_validate(self.model, x, y, **self.cv_params)
        else:
            if val_y is not None:
                self.model = self.model.fit(x, y, val_x, val_y)
            else:
                self.model = self.model.fit(x, y)

        return self.model, training_results


class MultiColumnsTrainer(Trainer):
    def __init__(self, model_blueprint, conf, queue=None):
        self.preds = {}

        self.model_blueprint = model_blueprint
        self.conf = conf

        self.queue = queue

        self.run_cross_validation = self.conf['train']['run_cross_validation']
        self.cv_params = conf['train']['cv_params'].copy()

    def train(self, x, y, val_x=None, val_y=None):
        training_results = {}
        result_tickets = {}

        trained_cols = []

        for col in tqdm(y.columns, desc="Sending jobs"):
            self.preds[col] = deepcopy(self.model_blueprint)

            col_trainer = SingleColumnTrainer(self.preds[col],
                                              conf=self.conf)

            if self.queue is not None:
                if val_y is not None:
                    result_tickets[col] = self.queue.method(col_trainer.train, (x, y[col], val_x, val_y[col]))
                else:
                    result_tickets[col] = self.queue.method(col_trainer.train, (x, y[col]))
            else:
                if val_y is not None:
                    self.preds[col], training_results[col] = col_trainer.train(x, y[col], val_x, val_y[col])
                else:
                    self.preds[col], training_results[col] = col_trainer.train(x, y[col])

        if self.queue is not None:
            for col in tqdm(y.columns, desc="Fetching results"):
                self.preds[col], training_results[col] = self.queue.waitforresult(result_tickets[col])
                trained_cols += col

        return self.preds, training_results


def train_multi_model(train_x, train_y, trainable: Trainable, conf: dict, queue=None, val_x=None, val_y=None):
    trainer = MultiColumnsTrainer(
        trainable,
        queue=queue,
        conf=conf
    )

    models, training_results = trainer.train(train_x, train_y, val_x, val_y)

    print('multi training finished')

    return models, training_results
