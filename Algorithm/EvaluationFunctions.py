import chess
import chess.engine
import math
import copy


'''All evaluation functions take in a "player" parameter and return a positive number corresponding to how good
 that player's position is. If you need to compare it to the position of the other player or to find out
  which player currently has an advantage, call the evaluation function being used twice, once for each player's
   perspective and then compare them.'''


#simple evaluation function which improves on the one from the tutorial we found in minmax.py by making a bishop
# worth slightly more than a knight, and also giving the king a value to encourage capturing the king for a checkmate.
#https://www.chessprogramming.org/Simplified_Evaluation_Function
def simple_material_evaluator(BOARD, player='white'):
    scoring = {'P': 100,
               'N': 320,
               'B': 330,
               'R': 500,
               'Q': 900,
               'K': 20000,
               }

    score = 0
    pieces = BOARD.piece_map()
    for key in pieces:
        if str(pieces[key]).isupper() and player == 'white':
            score += scoring[str(pieces[key])]
        elif str(pieces[key]).islower() and player == 'black':
            score += scoring[str(pieces[key]).upper()]

    return score


#uses stockfish's evaluator in order to test our algorithms with a known good evaluator
#https://stackoverflow.com/questions/58556338/python-evaluating-a-board-position-using-stockfish-from-the-python-chess-librar
def stockfish_evaluator(BOARD, player='white'):
    engine = chess.engine.SimpleEngine.popen_uci("stockfish")
    result = engine.analyse(BOARD, chess.engine.Limit(time=0.01))
    if player == 'white':
        return result['score'].white()
    else:
        return result['score'].black()





'''An evaluator we made that takes into account the value of the positions of each piece in order to encourage positional play.
It works by using Piece-Square tables, which assign specific values for each square on the board per piece type. We then
combine the values across all pieces at their current positions to evaluate the total value for the position. 
The Piece-Square tables originat here: https://www.chessprogramming.org/Simplified_Evaluation_Function

One custom improvement made was adding the material value of the piece positional value when computing the score.
The reason for this is that otherwise it could appear better to let a piece be captured then have it occupy
a bad square, but we don't want our chess engine giving away pieces that have negative positional value.
this way we ensure that all values are still positive, and also account for the different worth of each piece'''

def basic_piece_squares_and_material_evaluator(BOARD, player='white'):
    scoring = {'P': 100,
               'N': 320,
               'B': 330,
               'R': 500,
               'Q': 900,
               'K': 20000,
               }

    pawn = [ [0,  0,  0,  0,  0,  0,  0,  0],
            [50, 50, 50, 50, 50, 50, 50, 50],
            [10, 10, 20, 30, 30, 20, 10, 10],
             [5,  5, 10, 25, 25, 10,  5,  5],
             [0,  0,  0, 20, 20,  0,  0,  0],
             [5, -5,-10,  0,  0,-10, -5,  5],
             [5, 10, 10,-20,-20, 10, 10,  5],
             [0,  0,  0,  0,  0,  0,  0,  0]]

    knight = [  [-50,-40,-30,-30,-30,-30,-40,-50],
                [-40,-20,  0,  0,  0,  0,-20,-40],
                [-30,  0, 10, 15, 15, 10,  0,-30],
                [-30,  5, 15, 20, 20, 15,  5,-30],
                [-30,  0, 15, 20, 20, 15,  0,-30],
                [-30,  5, 10, 15, 15, 10,  5,-30],
                [-40,-20,  0,  5,  5,  0,-20,-40],
                [-50,-40,-30,-30,-30,-30,-40,-50]]

    bishop = [  [-20,-10,-10,-10,-10,-10,-10,-20],
                [-10,  0,  0,  0,  0,  0,  0,-10],
                [-10,  0,  5, 10, 10,  5,  0,-10],
                [-10,  5,  5, 10, 10,  5,  5,-10],
                [-10,  0, 10, 10, 10, 10,  0,-10],
                [-10, 10, 10, 10, 10, 10, 10,-10],
                [-10,  5,  0,  0,  0,  0,  5,-10],
                [-20,-10,-10,-10,-10,-10,-10,-20]]

    rook = [  [0,  0,  0,  0,  0,  0,  0,  0],
              [5, 10, 10, 10, 10, 10, 10,  5],
             [-5,  0,  0,  0,  0,  0,  0, -5],
             [-5,  0,  0,  0,  0,  0,  0, -5],
             [-5,  0,  0,  0,  0,  0,  0, -5],
             [-5,  0,  0,  0,  0,  0,  0, -5],
             [-5,  0,  0,  0,  0,  0,  0, -5],
              [0,  0,  0,  5,  5,  0,  0,  0]]

    queen = [   [-20,-10,-10, -5, -5,-10,-10,-20],
                [-10,  0,  0,  0,  0,  0,  0,-10],
                [-10,  0,  5,  5,  5,  5,  0,-10],
                [-5,  0,  5,  5,  5,  5,  0, -5],
                  [0,  0,  5,  5,  5,  5,  0, -5],
                [-10,  5,  5,  5,  5,  5,  0,-10],
                [-10,  0,  5,  0,  0,  0,  0,-10],
                [-20,-10,-10, -5, -5,-10,-10,-20]]

    king_middle_game = [[-30,-40,-40,-50,-50,-40,-40,-30],
                        [-30,-40,-40,-50,-50,-40,-40,-30],
                        [-30,-40,-40,-50,-50,-40,-40,-30],
                        [-30,-40,-40,-50,-50,-40,-40,-30],
                        [-20,-30,-30,-40,-40,-30,-30,-20],
                        [-10,-20,-20,-20,-20,-20,-20,-10],
                        [20, 20,  0,  0,  0,  0, 20, 20],
                         [20, 30, 10,  0,  0, 10, 30, 20]]

    king_end_game = [   [-50,-40,-30,-20,-20,-30,-40,-50],
                        [-30,-20,-10,  0,  0,-10,-20,-30],
                        [-30,-10, 20, 30, 30, 20,-10,-30],
                        [-30,-10, 30, 40, 40, 30,-10,-30],
                        [-30,-10, 30, 40, 40, 30,-10,-30],
                        [-30,-10, 20, 30, 30, 20,-10,-30],
                        [-30,-30,  0,  0,  0,  0,-30,-30],
                        [-50,-30,-30,-30,-30,-30,-30,-50]]

    score = 0
    if player == 'white':
        pieces = BOARD.piece_map()
    else:
        pieces = BOARD.mirror().piece_map()

    if len(pieces) < 8:
        isEndGame = True
    else:
        isEndGame = False

    for key in pieces:
        piece = str(pieces[key])
        outerKey = 7 - math.floor(key / 8)
        innerKey = key % 8


        if piece.upper() :
            if piece == 'P':
                score += pawn[outerKey][innerKey] + scoring['P']
            elif piece == 'R':
                score += rook[outerKey][innerKey] + scoring['R']
            elif piece == 'N':
                score += knight[outerKey][innerKey] + scoring['N']
            elif piece == 'B':
                score += bishop[outerKey][innerKey] + scoring['B']
            elif piece == 'K':
                if isEndGame:
                    score += king_end_game[outerKey][innerKey] + scoring['K']
                else:
                    score += king_middle_game[outerKey][innerKey] + scoring['K']
            elif piece == 'Q':
                score += queen[key]
            else:
                raise Exception("Invalid piece in basic_piece_squares_evaluator")

    return score



'''fen = "8/6p1/8/8/8/8/8/8"
BOARD = chess.Board(fen)

print(basic_piece_squares_evaluator(BOARD, player='black'))'''

