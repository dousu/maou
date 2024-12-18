import abc


class Parser(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def parse(self, content: str) -> None:
        pass

    @abc.abstractmethod
    def init_pos_sfen(self) -> str:
        pass

    @abc.abstractmethod
    def endgame(self) -> str:
        pass

    @abc.abstractmethod
    def winner(self) -> int:
        pass

    @abc.abstractmethod
    def ratings(self) -> list[int]:
        pass

    @abc.abstractmethod
    def moves(self) -> list[str]:
        pass

    @abc.abstractmethod
    def scores(self) -> list[int]:
        pass

    @abc.abstractmethod
    def comments(self) -> list[str]:
        pass
