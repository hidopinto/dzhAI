from abc import abstractmethod

from backbone.train_utils.trainable import Trainable


class TrainableFactory:
    @abstractmethod
    def create(self) -> Trainable:
        pass
