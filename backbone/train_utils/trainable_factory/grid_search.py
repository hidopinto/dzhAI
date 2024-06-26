from backbone.train_utils.trainable import Trainable,  GridSearchTrainable
from backbone.train_utils.trainable_factory.trainable_factory import TrainableFactory


class GridSearchTrainableFactory(TrainableFactory):
    def __init__(self, model_blueprint, conf):
        self.model_blueprint = model_blueprint
        self.conf = conf

    def create(self) -> Trainable:
        return GridSearchTrainable(self.model_blueprint, **self.conf['train']['trainable']['args'])
