from src.engine.search import NegamaxSearcher


class EnginePlayer:
    def __init__(self, evaluator, depth=2):
        self.searcher = NegamaxSearcher(evaluator, depth)

    def select_move(self, board):
        return self.searcher.search(board)
