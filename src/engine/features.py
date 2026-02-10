import chess
import numpy as np


PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}


class FeatureExtractor:
    """
    Extracts a fixed-length feature vector f(board).
    All features are from White's perspective.
    """

    def extract(self, board: chess.Board) -> np.ndarray:
        features = []

        # --- Material counts ---
        for piece_type in PIECE_VALUES:
            features.append(
                len(board.pieces(piece_type, chess.WHITE))
                - len(board.pieces(piece_type, chess.BLACK))
            )

        # --- Bishop pair ---
        features.append(
            int(len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2)
            - int(len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2)
        )

        # --- Mobility ---
        features.append(self._mobility(board))

        # --- Pawn structure ---
        features.extend(self._pawn_structure(board))

        # --- King safety ---
        features.append(self._king_safety(board))

        # --- Pawn advancement ---
        features.append(self._pawn_advancement(board))

        return np.array(features, dtype=np.float32)

    def _mobility(self, board: chess.Board) -> float:
        turn = board.turn
        board.turn = chess.WHITE
        white_moves = board.legal_moves.count()
        board.turn = chess.BLACK
        black_moves = board.legal_moves.count()
        board.turn = turn
        return white_moves - black_moves

    def _pawn_structure(self, board: chess.Board):
        def count_isolated(color):
            pawns = board.pieces(chess.PAWN, color)
            files = [chess.square_file(sq) for sq in pawns]
            isolated = 0
            for sq in pawns:
                f = chess.square_file(sq)
                if f - 1 not in files and f + 1 not in files:
                    isolated += 1
            return isolated

        def count_doubled(color):
            pawns = board.pieces(chess.PAWN, color)
            files = [chess.square_file(sq) for sq in pawns]
            return len(files) - len(set(files))

        return [
            count_isolated(chess.WHITE) - count_isolated(chess.BLACK),
            count_doubled(chess.WHITE) - count_doubled(chess.BLACK),
        ]

    def _king_safety(self, board: chess.Board):
        score = 0
        for color, sign in [(chess.WHITE, 1), (chess.BLACK, -1)]:
            king_sq = board.king(color)
            attackers = board.attackers(not color, king_sq)
            score -= sign * len(attackers)
        return score

    def _pawn_advancement(self, board: chess.Board):
        score = 0
        for sq in board.pieces(chess.PAWN, chess.WHITE):
            score += chess.square_rank(sq)
        for sq in board.pieces(chess.PAWN, chess.BLACK):
            score -= 7 - chess.square_rank(sq)
        return score
