import chess
from Algorithm.EvaluationFunctions import *

class NegaMax:
    def __init__(self, search_depth=6, evaluation_function=simple_material_evaluator, player='white'):
        self.search_depth = search_depth
        self.evaluation_function = evaluation_function
        self.player = player

    def run_negamax(self, current_board, depth_remaining, alpha, beta):
        if depth_remaining == 0 or current_board.is_game_over():
            return self.evaluation_function(current_board, self.player)

        maximum_evaluation = float('-inf')
        for potential_move in current_board.legal_moves:
            current_board.push(potential_move)
            evaluation = -self.run_negamax(current_board, depth_remaining - 1, -beta, -alpha)
            current_board.pop()
            maximum_evaluation = max(maximum_evaluation, evaluation)
            alpha = max(alpha, evaluation)
            if alpha >= beta:
                break
        return maximum_evaluation

    def find_best_move(self, board):
        optimal_move = None
        highest_value = float('-inf')
        for move in board.legal_moves:
            board.push(move)
            move_evaluation = -self.run_negamax(board, self.search_depth - 1, float('-inf'), float('inf'))
            board.pop()
            if move_evaluation > highest_value:
                highest_value = move_evaluation
                optimal_move = move
                return optimal_move

    
def play_nega_max(current_board, search_depth=3, evaluation_function=relative_pieces_attacked_evaluator, player='white'):
    negamax = NegaMax(search_depth, evaluation_function, player)
    return negamax.find_best_move(current_board)
