import chess
import chess.engine
from copy import deepcopy

import chess

class MinMax:
    def __init__(self, evaluation_function):
        self.evaluation_function = evaluation_function

    def min_max(self, board, depth, player_color):
        if depth == 0 or board.is_game_over():
            return self.evaluation_function(board, player_color), None

        moves = list(board.legal_moves)
        best_score = float('-inf') if board.turn == chess.WHITE else float('inf')
        best_move = None

        for move in moves:
            temp_board = deepcopy(board)
            temp_board.push(move)
            next_player_color = "black" if player_color == "white" else "white"
            score, _ = self.min_max(temp_board, depth - 1, next_player_color)

            if board.turn == chess.WHITE:
                if score > best_score:
                    best_score = score
                    best_move = move
            else:
                if score < best_score:
                    best_score = score
                    best_move = move

        return best_score, best_move

    def select_move(self, board, depth=3):
        player_color = "white" if board.turn == chess.WHITE else "black"
        _, best_move = self.min_max(board, depth, player_color)
        return best_move

def play_min_max(evaluator):
    return MinMax(evaluator)
