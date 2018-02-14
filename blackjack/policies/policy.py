from abc import ABC, abstractmethod

class Policy(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        ...


