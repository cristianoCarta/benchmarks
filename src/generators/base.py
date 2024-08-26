from abc import ABC


class SampleGenerator(ABC):

    def create_dataset(self, 
                       N : int,
                       name : str = None):
        raise NotImplementedError("implement me")