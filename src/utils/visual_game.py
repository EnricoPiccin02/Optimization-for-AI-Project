import chess
import tkinter as tk
import time

# Unicode chess symbols
PIECE_SYMBOLS = {
    chess.PAWN: ("♙", "♟"),
    chess.ROOK: ("♖", "♜"),
    chess.KNIGHT: ("♘", "♞"),
    chess.BISHOP: ("♗", "♝"),
    chess.QUEEN: ("♕", "♛"),
    chess.KING: ("♔", "♚"),
}


class VisualGame:
    """
    Graphical chessboard visualisation using Tkinter Canvas
    and Unicode chess symbols.

    Designed for qualitative inspection of trained agents.
    """

    SQUARE_SIZE = 80
    LIGHT_COLOR = "#F0D9B5"
    DARK_COLOR = "#B58863"

    def __init__(self, white, black, delay=0.5, title="Chess AI Match"):
        self.white = white
        self.black = black
        self.delay = delay
        self.board = chess.Board()

        self.root = tk.Tk()
        self.root.title(title)

        size = self.SQUARE_SIZE * 8
        self.canvas = tk.Canvas(self.root, width=size, height=size)
        self.canvas.pack()

        self._draw_board()

    def _draw_board(self):
        self.canvas.delete("all")

        for rank in range(8):
            for file in range(8):
                x1 = file * self.SQUARE_SIZE
                y1 = rank * self.SQUARE_SIZE
                x2 = x1 + self.SQUARE_SIZE
                y2 = y1 + self.SQUARE_SIZE

                color = self.LIGHT_COLOR if (rank + file) % 2 == 0 else self.DARK_COLOR

                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")

                square = chess.square(file, 7 - rank)
                piece = self.board.piece_at(square)

                if piece:
                    symbol = PIECE_SYMBOLS[piece.piece_type][
                        0 if piece.color == chess.WHITE else 1
                    ]
                    self.canvas.create_text(
                        x1 + self.SQUARE_SIZE / 2,
                        y1 + self.SQUARE_SIZE / 2,
                        text=symbol,
                        font=("Arial", 48),
                    )

        self.root.update_idletasks()
        self.root.update()

    def play(self, max_moves=200):
        move_count = 0
        self._draw_board()

        while not self.board.is_game_over() and move_count < max_moves:
            player = self.white if self.board.turn == chess.WHITE else self.black

            move = player.select_move(self.board)
            if move is None:
                break

            self.board.push(move)
            self._draw_board()
            time.sleep(self.delay)
            move_count += 1

        print("Game result:", self.board.result())
        self.root.mainloop()
