import chess

class Negamax:
    def __init__(self, search_depth=6):
        self.search_depth = search_depth

    def run_negamax(self, current_board, depth_remaining, alpha, beta, player_color):
        if depth_remaining == 0 or current_board.is_game_over():
            return player_color * self.board_evaluation(current_board)

        maximum_evaluation = float('-inf')
        for potential_move in current_board.legal_moves:
            current_board.push(potential_move)
            evaluation = -self.run_negamax(current_board, depth_remaining - 1, -beta, -alpha, -player_color)
            current_board.pop()
            maximum_evaluation = max(maximum_evaluation, evaluation)
            alpha = max(alpha, evaluation)
            ## alpha-beta pruning occurs
            if alpha >= beta:
                break

        return maximum_evaluation

    def board_evaluation(self, board):
        piece_value_map = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                           chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
        total_score = 0
        for piece in piece_value_map.keys():
            total_score += len(board.pieces(piece, chess.WHITE)) * piece_value_map[piece]
            total_score -= len(board.pieces(piece, chess.BLACK)) * piece_value_map[piece]
        return total_score

    def find_best_move(self, board):
        optimal_move = None
        highest_value = float('-inf')
        for move in board.legal_moves:
            board.push(move)
            move_evaluation = -self.run_negamax(board, self.search_depth - 1, float('-inf'), float('inf'), -1)
            board.pop()
            if move_evaluation > highest_value:
                highest_value = move_evaluation
                optimal_move = move
        return optimal_move

def play_nega_max(current_board, search_depth=5):
    negamax = Negamax(search_depth)
    return negamax.find_best_move(current_board)

