import chess


def play_game(white_player, black_player, max_moves=200, verbose=False):
    board = chess.Board()
    move_count = 0

    while not board.is_game_over() and move_count < max_moves:
        if board.turn == chess.WHITE:
            move = white_player.select_move(board)
        else:
            move = black_player.select_move(board)

        if move is None:
            break

        board.push(move)
        move_count += 1

        if verbose:
            print(board)
            print()

    result = board.result(claim_draw=True)
    return result
