

import chess

class NegaMax:
    def __init__(self):
        # Scoring dictionary for basic evaluation
        self.scoring = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
        }

    def negamax(self, board, depth, alpha, beta, color):
        # Base case: depth is 0 or game is over
        if depth == 0 or board.is_game_over():
            return color * self.evaluate(board)

        max_value = float('-inf')
        for move in board.legal_moves:
            board.push(move)
            value = -self.negamax(board, depth - 1, -beta, -alpha, -color)
            board.pop()
            max_value = max(max_value, value)
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return max_value

    def evaluate(self, board):
        # Simple evaluation function based on material
        score = 0
        for piece_type in self.scoring:
            score += len(board.pieces(piece_type, chess.WHITE)) * self.scoring[piece_type]
            score -= len(board.pieces(piece_type, chess.BLACK)) * self.scoring[piece_type]
        return score

    def select_move(self, board, depth):
        best_move = None
        best_value = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        for move in board.legal_moves:
            board.push(move)
            value = -self.negamax(board, depth - 1, -beta, -alpha, 1)
            board.pop()
            if value > best_value:
                best_value = value
                best_move = move
        return best_move
def play_nega_max(board):
    negamax = NegaMax()
    return negamax.select_move(board)