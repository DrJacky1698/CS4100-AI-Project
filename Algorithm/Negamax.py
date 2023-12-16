import chess
from Algorithm.EvaluationFunctions import *

class NegaMax:
    def __init__(self, evaluation_function=simple_material_evaluator, search_depth=3):
        self.search_depth = search_depth
        self.evaluation_function = evaluation_function

    def run_negamax(self, current_board, depth_remaining, alpha, beta):
        player_color = "white" if current_board.turn == chess.WHITE else "black"
        if depth_remaining == 0 or current_board.is_game_over():
            return self.evaluation_function(current_board, player_color)

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

    def select_move(self, board):
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

    
def play_nega_max(evaluation_function=relative_pieces_attacked_evaluator, search_depth=3):
    # negamax = NegaMax(evaluation_function, search_depth)
    # return negamax.select_move(current_board)
    return NegaMax(evaluation_function, search_depth)
