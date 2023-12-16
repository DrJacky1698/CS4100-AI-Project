import chess
from Algorithm.EvaluationFunctions import *

class NegaMax:
    # initialize NegaMax with an evaluation function and search depth
    def __init__(self, evaluation_function=simple_material_evaluator, search_depth=3):
        self.search_depth = search_depth
        self.evaluation_function = evaluation_function

    # Negamax algorithm with alpha-beta pruning
    def run_negamax(self, current_board, depth_remaining, alpha, beta):
        player_color = "white" if current_board.turn == chess.WHITE else "black"
        # return the board state if max depth reached or game over
        if depth_remaining == 0 or current_board.is_game_over():
            return self.evaluation_function(current_board, player_color)

        maximum_evaluation = float('-inf')
        # explore possible moves and go recursive on Negamax
        for potential_move in current_board.legal_moves:
            current_board.push(potential_move)
            evaluation = -self.run_negamax(current_board, depth_remaining - 1, -beta, -alpha)
            current_board.pop()
            maximum_evaluation = max(maximum_evaluation, evaluation)
            alpha = max(alpha, evaluation)
            # prune the branch if alpha is greater or equal to beta
            if alpha >= beta:
                break
        return maximum_evaluation

    # select the best move based on Negamax algorithm
    def select_move(self, board):
        optimal_move = None
        highest_value = float('-inf')
        # evaluate each possible move to find the best
        for move in board.legal_moves:
            board.push(move)
            move_evaluation = -self.run_negamax(board, self.search_depth - 1, float('-inf'), float('inf'))
            board.pop()
            # update the best move
            if move_evaluation > highest_value:
                highest_value = move_evaluation
                optimal_move = move
        return optimal_move

def play_nega_max(evaluation_function=relative_pieces_attacked_evaluator, search_depth=3):
    # create an instance and return the selected move
    return NegaMax(evaluation_function, search_depth)
