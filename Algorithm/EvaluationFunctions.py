import chess
import chess.engine
import math
import copy

engine = False

'''Use the relative evaluators instead. The relative .......'''

'''All evaluation functions take in a "player" parameter and return a number corresponding to how good
 that player's position is. If you need to compare it to the position of the other player or to find out
  which player currently has an advantage, call the evaluation function being used twice, once for each player's
   perspective and then compare them.'''


#simple evaluation function which improves on the one from the tutorial we found in minmax.py by making a bishop
# worth slightly more than a knight, and also giving the king a value to encourage capturing the king for a checkmate.
#https://www.chessprogramming.org/Simplified_Evaluation_Function
def simple_material_evaluator(BOARD, player):
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


def relative_simple_material_evaluator(BOARD, player):
    if player == 'white':
        return simple_material_evaluator(BOARD, player='white') - simple_material_evaluator(BOARD, player='black')
    else:
        return simple_material_evaluator(BOARD, player='black') - simple_material_evaluator(BOARD, player='white')

def initializeStockfishEngine():
    global engine
    #engine = chess.engine.SimpleEngine.popen_uci("Algorithm/stockfish_simon")
    engine = chess.engine.SimpleEngine.popen_uci("stockfish")


def closeStockfishEngine():
    global engine
    engine.close()



#uses stockfish's evaluator in order to test our algorithms with a known good evaluator
#https://stackoverflow.com/questions/58556338/python-evaluating-a-board-position-using-stockfish-from-the-python-chess-librar
def stockfish_evaluator(BOARD, player):
    #engine = chess.engine.SimpleEngine.popen_uci("Algorithm/stockfish_simon")
    result = engine.analyse(BOARD, chess.engine.Limit(time=0.01))
    #engine.close()
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

def basic_piece_squares_and_material_evaluator(BOARD, player):
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


        if piece.isupper() :
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


def relative_basic_piece_squares_and_material_evaluator(BOARD, player):
    if player == 'white':
        return basic_piece_squares_and_material_evaluator(BOARD, player='white') - basic_piece_squares_and_material_evaluator(BOARD, player='black')
    else:
        return basic_piece_squares_and_material_evaluator(BOARD, player='black') - basic_piece_squares_and_material_evaluator(BOARD, player='white')

'''This evaluator improves on basic_piece_squares_and_material_evaluator in several ways ways.
Firstly, it features different values for all the piece types from the middle game to the end game, which
causes better strategical positioning during different phases of the game. Two, it features tapered evaluation, meaning
it blends the values it assigns the current position of each piece in a gradual manner as the game progresses, rather
than abruptly switching between the middle game and end game positional evaluations suddenly. Lastly, from our research
we believe that the values in these tables are overall higher quality then the ones from basic_piece_squares_and_material_evaluator, 
and so should lead to better performance. We sourced these values from the following link, but the code for using them is all our own
and is based on expanding what we did for the basic_piece_squares_and_material_evaluator.
https://www.chessprogramming.org/PeSTO%27s_Evaluation_Function
'''
def tapered_piece_squares_evaluator(BOARD, player):
    mg_pawn_table = [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [98, 134, 61, 95, 68, 126, 34, -11],
        [-6, 7, 26, 31, 65, 56, 25, -20],
        [-14, 13, 6, 21, 23, 12, 17, -23],
        [-27, -2, -5, 12, 17, 6, 10, -25],
        [-26, -4, -4, -10, 3, 3, 33, -12],
        [-35, -1, -20, -23, -15, 24, 38, -22],
        [0, 0, 0, 0, 0, 0, 0, 0]]


    eg_pawn_table = [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [178, 173, 158, 134, 147, 132, 165, 187],
        [94, 100, 85, 67, 56, 53, 82, 84],
        [32, 24, 13, 5, -2, 4, 17, 17],
        [13, 9, -3, -7, -7, -8, 3, -1],
        [4, 7, -6, 1, 0, -5, -1, -8],
        [13, 8, 8, 10, 13, 0, 2, -7],
        [0, 0, 0, 0, 0, 0, 0, 0]]


    mg_knight_table = [
        [-167, -89, -34, -49, 61, -97, -15, -107],
        [-73, -41, 72, 36, 23, 62, 7, -17],
        [-47, 60, 37, 65, 84, 129, 73, 44],
        [-9, 17, 19, 53, 37, 69, 18, 22],
        [-13, 4, 16, 13, 28, 19, 21, -8],
        [-23, -9, 12, 10, 19, 17, 25, -16],
        [-29, -53, -12, -3, -1, 18, -14, -19],
        [-105, -21, -58, -33, -17, -28, -19, -23]]



    eg_knight_table = [
        [-58, -38, -13, -28, -31, -27, -63, -99],
        [-25, -8, -25, -2, -9, -25, -24, -52],
        [-24, -20, 10, 9, -1, -9, -19, -41],
        [-17, 3, 22, 22, 22, 11, 8, -18],
        [-18, -6, 16, 25, 16, 17, 4, -18],
        [-23, -3, -1, 15, 10, -3, -20, -22],
        [-42, -20, -10, -5, -2, -20, -23, -44],
        [-29, -51, -23, -15, -22, -18, -50, -64]]


    mg_bishop_table = [
        [-29, 4, -82, -37, -25, -42, 7, -8],
        [-26, 16, -18, -13, 30, 59, 18, -47],
        [-16, 37, 43, 40, 35, 50, 37, -2],
        [-4, 5, 19, 50, 37, 37, 7, -2],
        [-6, 13, 13, 26, 34, 12, 10, 4],
        [0, 15, 15, 15, 14, 27, 18, 10],
        [4, 15, 16, 0, 7, 21, 33, 1],
        [-33, -3, -14, -21, -13, -12, -39, -21]]

    eg_bishop_table = [
        [-14, -21, -11, -8, -7, -9, -17, -24],
        [-8, -4, 7, -12, -3, -13, -4, -14],
        [2, -8, 0, -1, -2, 6, 0, 4],
        [-3, 9, 12, 9, 14, 10, 3, 2],
        [-6, 3, 13, 19, 7, 10, -3, -9],
        [-12, -3, 8, 10, 13, 3, -7, -15],
        [-14, -18, -7, -1, 4, -9, -15, -27],
        [-23, -9, -23, -5, -9, -16, -5, -17]
    ]

    mg_rook_table = [
        [32, 42, 32, 51, 63, 9, 31, 43],
        [27, 32, 58, 62, 80, 67, 26, 44],
        [-5, 19, 26, 36, 17, 45, 61, 16],
        [-24, -11, 7, 26, 24, 35, -8, -20],
        [-36, -26, -12, -1, 9, -7, 6, -23],
        [-45, -25, -16, -17, 3, 0, -5, -33],
        [-44, -16, -20, -9, -1, 11, -6, -71],
        [-19, -13, 1, 17, 16, 7, -37, -26]
    ]

    eg_rook_table = [
        [13, 10, 18, 15, 12, 12, 8, 5],
        [11, 13, 13, 11, -3, 3, 8, 3],
        [7, 7, 7, 5, 4, -3, -5, -3],
        [4, 3, 13, 1, 2, 1, -1, 2],
        [3, 5, 8, 4, -5, -6, -8, -11],
        [-4, 0, -5, -1, -7, -12, -8, -16],
        [-6, -6, 0, 2, -9, -9, -11, -3],
        [-9, 2, 3, -1, -5, -13, 4, -20]
    ]

    mg_queen_table = [
        [-28, 0, 29, 12, 59, 44, 43, 45],
        [-24, -39, -5, 1, -16, 57, 28, 54],
        [-13, -17, 7, 8, 29, 56, 47, 57],
        [-27, -27, -16, -16, -1, 17, -2, 1],
        [-9, -26, -9, -10, -2, -4, 3, -3],
        [-14, 2, -11, -2, -5, 2, 14, 5],
        [-35, -8, 11, 2, 8, 15, -3, 1],
        [-1, -18, -9, 10, -15, -25, -31, -50]
    ]

    eg_queen_table = [
        [-9, 22, 22, 27, 27, 19, 10, 20],
        [-17, 20, 32, 41, 58, 25, 30, 0],
        [-20, 6, 9, 49, 47, 35, 19, 9],
        [3, 22, 24, 45, 57, 40, 57, 36],
        [-18, 28, 19, 47, 31, 34, 39, 23],
        [-16, -27, 15, 6, 9, 17, 10, 5],
        [-22, -23, -30, -16, -16, -23, -36, -32],
        [-33, -28, -22, -43, -5, -32, -20, -41]
    ]

    mg_king_table = [
        [-65, 23, 16, -15, -56, -34, 2, 13],
        [29, -1, -20, -7, -8, -4, -38, -29],
        [-9, 24, 2, -16, -20, 6, 22, -22],
        [-17, -20, -12, -27, -30, -25, -14, -36],
        [-49, -1, -27, -39, -46, -44, -33, -51],
        [-14, -14, -22, -46, -44, -30, -15, -27],
        [1, 7, -8, -64, -43, -16, 9, 8],
        [-15, 36, 12, -54, 8, -28, 24, 14]
    ]

    eg_king_table = [
        [-74, -35, -18, -18, -11, 15, 4, -17],
        [-12, 17, 14, 17, 17, 38, 23, 11],
        [10, 17, 23, 15, 20, 45, 44, 13],
        [-8, 22, 24, 27, 26, 33, 26, 3],
        [-18, -4, 21, 24, 27, 23, 9, -11],
        [-19, -3, 11, 21, 23, 16, 7, -9],
        [-27, -11, 4, 13, 14, 4, -5, -17],
        [-53, -34, -21, -11, -28, -14, -24, -43]
    ]

    score = 0
    if player == 'white':
        pieces = BOARD.piece_map()
    else:
        pieces = BOARD.mirror().piece_map()

    middleGameModifier = len(pieces) / 32
    endGameModifier = (32 - len(pieces)) / 32


    for key in pieces:
        piece = str(pieces[key])
        outerKey = 7 - math.floor(key / 8)
        innerKey = key % 8

        if piece.isupper():
            if piece == 'P':
                score += mg_pawn_table[outerKey][innerKey] * middleGameModifier + eg_pawn_table[outerKey][innerKey] * endGameModifier
            elif piece == 'R':
                score += mg_rook_table[outerKey][innerKey] * middleGameModifier + eg_rook_table[outerKey][innerKey] * endGameModifier
            elif piece == 'N':
                score += mg_knight_table[outerKey][innerKey] * middleGameModifier + eg_knight_table[outerKey][innerKey] * endGameModifier
            elif piece == 'B':
                score += mg_bishop_table[outerKey][innerKey] * middleGameModifier + eg_bishop_table[outerKey][innerKey] * endGameModifier
            elif piece == 'K':
                score += mg_king_table[outerKey][innerKey] * middleGameModifier + eg_king_table[outerKey][innerKey] * endGameModifier
            elif piece == 'Q':
                score += mg_queen_table[outerKey][innerKey] * middleGameModifier + eg_queen_table[outerKey][innerKey] * endGameModifier
            else:
                raise Exception("Invalid piece in basic_piece_squares_evaluator")

    return score


def relative_tapered_piece_squares_evaluator(BOARD, player='white'):
    if player == 'white':
        return tapered_piece_squares_evaluator(BOARD, player='white') - tapered_piece_squares_evaluator(BOARD, player='black')
    else:
        return tapered_piece_squares_evaluator(BOARD, player='black') - tapered_piece_squares_evaluator(BOARD, player='white')


'''This evaluator only evaluates the king safety of the provided side by looking at how many pieces the opponent has attacking the king or
any of the squares immediately adjacent to the king. If an enemy piece attacks multiple squares in this area, it will be counted multiple
times. Also, additional weight is placed on pieces attacking the king directly. In order to stay consistent with
the other evaluation function's, where higher output results are better then lower results, the number of attacked squares
is returned as a negative number, since having more squares near the king under attack, means lower king safety,
 which is a bad thing.'''
def king_safety_evaluator(BOARD, player='white'):
    if player == 'white':
        playerColor = chess.WHITE
        enemyColor = chess.BLACK
    else:
        playerColor = chess.BLACK
        enemyColor = chess.WHITE

    playerKingSquare = BOARD.king(playerColor)

    enemyAttackers = 0
    playerKingAdjacentSquares = BOARD.attacks(playerKingSquare)
    for square in playerKingAdjacentSquares:
        enemyAttackers += len(BOARD.attackers(enemyColor, square))

    enemyAttackers += 2 * len(BOARD.attackers(enemyColor, playerKingSquare))

    return enemyAttackers * -1

def relative_king_safety_evaluator(BOARD, player='white'):
    if player == 'white':
        return king_safety_evaluator(BOARD, player='white') - king_safety_evaluator(BOARD, player='black')
    else:
        return king_safety_evaluator(BOARD, player='black') - king_safety_evaluator(BOARD, player='white')

'''This very simple evaluator just counts the amount of pieces present for the given side.
it is intended to be used as one of the evaluation functions for the random forest learning algorithm
in tandem with other more advanced evaluation functions.'''
def simple_piece_count_evaluator(BOARD, player='white'):
    score = 0
    if player == 'white':
        pieces = BOARD.piece_map()
    else:
        pieces = BOARD.mirror().piece_map()

    for key in pieces:
        piece = str(pieces[key])
        if piece.isupper():
            score += 1

    return score

def relative_simple_piece_count_evaluator(BOARD, player='white'):
    if player == 'white':
        return simple_piece_count_evaluator(BOARD, player='white') - simple_piece_count_evaluator(BOARD, player='black')
    else:
        return simple_piece_count_evaluator(BOARD, player='black') - simple_piece_count_evaluator(BOARD, player='white')

'''This evaluator looks at the number of pieces for the provided side that are under attack. Since having more of your pieces
under attack is a bad thing, the total amount of pieces under attack is returned as a negative number to stay consistent with
the format of the other evaluation functions where positive numbers denote a better position.Since this evaluator is intended
as part of the random forest training algorithm, to cause less overlap with other evaluators, any of the pieces that
 would be considered under attack by the king_safety_evaluator, will not also be considered under attack here to prevent 
 double-counting and evaluator overlap. This means that the king and all squares immediately around it will not be
counted towards the total number of squares being attacked for a side.'''
def pieces_attacked_evaluator(BOARD, player='white'):
    if player == 'white':
        playerColor = chess.WHITE
        enemyColor = chess.BLACK
    else:
        playerColor = chess.BLACK
        enemyColor = chess.WHITE

    enemyAttackers = 0

    pieces = BOARD.piece_map()

    playerKingSquare = BOARD.king(playerColor)
    playerKingAdjacentSquares = BOARD.attacks(playerKingSquare)

    for key in pieces:
        piece = str(pieces[key])
        if piece.isupper() and player == 'white' and key not in playerKingAdjacentSquares and key != playerKingSquare:
            enemyAttackers += len(BOARD.attackers(enemyColor, key))
        elif piece.islower() and player == 'black' and key not in playerKingAdjacentSquares and key != playerKingSquare:
            enemyAttackers += len(BOARD.attackers(enemyColor, key))

    return enemyAttackers * -1


def relative_pieces_attacked_evaluator(BOARD, player='white'):
    if player == 'white':
        return pieces_attacked_evaluator(BOARD, player='white') - pieces_attacked_evaluator(BOARD, player='black')
    else:
        return pieces_attacked_evaluator(BOARD, player='black') - pieces_attacked_evaluator(BOARD, player='white')

'''initializeStockfishEngine()
fen = "rnbqkbnr/1ppppppp/p7/8/8/6P1/PPPPPP1P/RNBQKBNR"
BOARD = chess.Board(fen)

print(tapered_piece_squares_evaluator(BOARD, player='white'))
print(tapered_piece_squares_evaluator(BOARD, player='black'))

print(relative_tapered_piece_squares_evaluator(BOARD, player='white'))
print(relative_tapered_piece_squares_evaluator(BOARD, player='black'))
closeStockfishEngine()'''


