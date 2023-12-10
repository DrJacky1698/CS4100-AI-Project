import chess
import chess.engine
from Algorithm.EvaluationFunctions import simple_material_evaluator
from copy import deepcopy

class MinMax:
    def __init__(self, evaluation_function):
        self.evaluation_function = evaluation_function

    def min_max(self, board, depth):
        if depth == 0 or board.is_game_over():
            return self.evaluation_function(board), None

        moves = list(board.legal_moves)
        best_score = float('-inf') if board.turn else float('inf')
        best_move = None

        for move in moves:
            temp_board = deepcopy(board)
            temp_board.push(move)
            score, _ = self.min_max(temp_board, depth - 1)

            if board.turn:
                if score > best_score:
                    best_score = score
                    best_move = move
            else:
                if score < best_score:
                    best_score = score
                    best_move = move

        return best_score, best_move

    def select_move(self, board, depth=3):
        _, best_move = self.min_max(board, depth)
        return best_move

def play_min_max(board):
    minmax = MinMax(simple_material_evaluator)
    return minmax.select_move(board)
