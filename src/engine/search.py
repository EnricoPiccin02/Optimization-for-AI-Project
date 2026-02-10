import chess
import random


MATE_SCORE = 10_000


class NegamaxSearcher:
    """
    Negamax search with alpha-beta pruning and stochastic tie-breaking
    among equally-valued root moves.

    Notes
    -----
    - A transposition table (TT) is used to speed up search.
    - The TT is cleared at the beginning of every root search to:
        * prevent unbounded memory growth
        * avoid cross-contamination between stochastic fitness evaluations
        * ensure independence of evolutionary objective samples
    """

    def __init__(self, evaluator, depth: int):
        self.evaluator = evaluator
        self.depth = depth
        self.tt = {}  # (fen, depth) -> value

    def search(self, board: chess.Board) -> chess.Move:
        """
        Select a move using negamax with alpha-beta pruning.

        The transposition table is cleared at each call to ensure
        bounded memory usage and evaluation independence.
        """
        self.tt.clear() # Clear TT at root search

        best_score = -float("inf")
        scored_moves = []

        for move in board.legal_moves:
            board.push(move)
            score = -self._negamax(
                board,
                self.depth - 1,
                -float("inf"),
                float("inf"),
            )
            board.pop()

            if score > best_score:
                best_score = score
                scored_moves = [(move, score)]
            elif score == best_score:
                scored_moves.append((move, score))

        # Stochastic tie-breaking among equally-valued moves
        return random.choice([m for m, _ in scored_moves])

    def _negamax(
        self,
        board: chess.Board,
        depth: int,
        alpha: float,
        beta: float,
    ) -> float:
        """
        Recursive negamax search with alpha-beta pruning.
        """
        key = (board.fen(), depth)
        if key in self.tt:
            return self.tt[key]

        if depth == 0 or board.is_game_over():
            value = self._evaluate_terminal(board)
            self.tt[key] = value
            return value

        value = -float("inf")

        for move in board.legal_moves:
            board.push(move)
            value = max(
                value,
                -self._negamax(board, depth - 1, -beta, -alpha),
            )
            board.pop()

            alpha = max(alpha, value)
            if alpha >= beta:
                break  # alpha-beta cutoff

        self.tt[key] = value
        return value

    def _evaluate_terminal(self, board: chess.Board) -> float:
        """
        Evaluate terminal or depth-limited positions.
        """
        if board.is_checkmate():
            return -MATE_SCORE
        if board.is_stalemate():
            return 0.0
        return self.evaluator.evaluate(board)
