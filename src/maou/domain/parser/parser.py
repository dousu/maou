import abc
from typing import Any


class Parser(metaclass=abc.ABCMeta):
    """Abstract base class for Shogi game record parsers.
    
    Defines the interface for parsing different Shogi game record formats
    (CSA, KIF, etc.) and extracting game information for training data.
    """
    @abc.abstractmethod
    def parse(self, content: str) -> None:
        """Parse game record content from string format.
        
        Args:
            content: Raw game record content as string
        """
        pass

    @abc.abstractmethod
    def init_pos_sfen(self) -> str:
        """Get initial board position in SFEN notation.
        
        Returns:
            Initial position as SFEN string
        """
        pass

    @abc.abstractmethod
    def endgame(self) -> str:
        """Get game termination reason.
        
        Returns:
            Endgame reason (e.g., 'TORYO', 'TIMEOUT', 'CHECKMATE')
        """
        pass

    @abc.abstractmethod
    def winner(self) -> int:
        """Get game winner.
        
        Returns:
            Winner player number (0 for first player, 1 for second player)
        """
        pass

    @abc.abstractmethod
    def ratings(self) -> list[int]:
        """Get player ratings.
        
        Returns:
            List of player ratings [first_player_rating, second_player_rating]
        """
        pass

    @abc.abstractmethod
    def moves(self) -> list[int]:
        """Get sequence of moves in the game.
        
        Returns:
            List of moves encoded as integers
        """
        pass

    @abc.abstractmethod
    def scores(self) -> list[int]:
        """Get evaluation scores for each position.
        
        Returns:
            List of position evaluation scores
        """
        pass

    @abc.abstractmethod
    def comments(self) -> list[str]:
        """Get comments for each move.
        
        Returns:
            List of move comments and annotations
        """
        pass

    @abc.abstractmethod
    def clustering_key_value(self) -> Any:
        """Get clustering key for data partitioning.
        
        Returns:
            Key value used for clustering in distributed storage
        """
        pass

    @abc.abstractmethod
    def partitioning_key_value(self) -> Any:
        """Get partitioning key for data distribution.
        
        Returns:
            Key value used for partitioning in distributed storage
        """
        pass
