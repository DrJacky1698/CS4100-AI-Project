import chess
import chess.polyglot
from copy import deepcopy

class MinMax:
    def __init__(self):
        self.scoring = {
            'p': -1, 'n': -3, 'b': -3, 'r': -5, 'q': -9, 'k': 0,
            'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 0,
        }

    def evaluate_board(self, board):
        score = 0
        pieces = board.piece_map()
        for key in pieces:
            score += self.scoring[str(pieces[key])]
        return score

    def evaluate_space(self, board):
        no_moves = len(list(board.legal_moves))
        value = no_moves / (20 + no_moves)
        return value if board.turn else -value

    def min_max(self, board, depth):
        if depth == 0 or board.is_game_over():
            return self.evaluate_board(board) + self.evaluate_space(board), None

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
    minmax = MinMax()
    return minmax.select_move(board)
