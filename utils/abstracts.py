from abc import ABC, abstractmethod


class AbsModel(ABC):
    @abstractmethod
    def get_val(self):
        pass

    @abstractmethod
    def learn(self):
        pass

    @abstractmethod
    def save_model(self):
        pass

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def _update(self):
        pass

    @abstractmethod
    def _compute_loss(self):
        pass

    @abstractmethod
    def __name__(self):
        pass


class AbsTrain(ABC):
    @abstractmethod
    def _step_agent(self):
        pass

    @abstractmethod
    def _step_env(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def save_model(self):
        pass


class AbsEval(ABC):
    @abstractmethod
    def _step_agent(self):
        pass

    @abstractmethod
    def _step_env(self):
        pass

    @abstractmethod
    def eval(self):
        pass

    @abstractmethod
    def load_model(self):
        pass


class AbsAsyncMaster(ABC):
    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def play(self):
        pass

