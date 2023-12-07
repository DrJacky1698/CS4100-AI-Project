import chess
import chess.polyglot
from copy import deepcopy

#Adding an opening book to compensate for the lack of depth the Min Max has
#reader = chess.polyglot.open_reader('Util/baron343-x64.exe')

scoring= {'p': -1,
          'n': -3,
          'b': -3,
          'r': -5,
          'q': -9,
          'k': 0,
          'P': 1,
          'N': 3,
          'B': 3,
          'R': 5,
          'Q': 9,
          'K': 0,
          }

#simple evaluation function
def eval_board(BOARD):
    score = 0
    pieces = BOARD.piece_map()
    for key in pieces:
        score += scoring[str(pieces[key])]

    return score

# secondary evaluation function
def eval_space(BOARD):
    no_moves = len(list(BOARD.legal_moves))

    #this function is always between 0 and 1 so we will never evaluate
    #this as being greater than a pawns value. The 20 value is arbitrary
    #but this number is chosen as it centers value around 0.5
    value = (no_moves/(20+no_moves))
    
    if BOARD.turn == True:
        return value
    else:
        return -value


def min_maxN(BOARD,N):

    '''#if move is found, return it as best move
    opening_move = reader.get(BOARD)
    if opening_move == None:
        pass
    else:
        return opening_move.move'''


    #The setting of letting game end not only gain more value
    
    #generate list of possible moves
    moves = list(BOARD.legal_moves)
    scores = []

    #score each move
    for move in moves:
        #temp allows us to leave the original game state unchanged
        temp = deepcopy(BOARD)
        temp.push(move)

        #here we must check that the game is not over
        outcome = temp.outcome()
        
        #if checkmate
        if outcome == None:
            #if we have not got to the final depth
            #we search more moves ahead
            if N>1:
                temp_best_move = min_maxN(temp,N-1)
                temp.push(temp_best_move)

            scores.append(eval_board(temp))

        #if checkmate
        elif temp.is_checkmate():

            # we return this as best move as it is checkmate
            return move

        # if stalemate
        else:
            #value to disencourage a draw
            #the higher the less likely to draw
            #default value should be 0
            #we often pick 0.1 to get the bot out of loops in bot vs bot
            val = 1000
            if BOARD.turn == True:
                scores.append(-val)
            else:
                scores.append(val)

        #this is the secondary eval function
        scores[-1] = scores[-1] + eval_space(temp)

    if BOARD.turn == True:
        best_move = moves[scores.index(max(scores))]
    else:
        best_move = moves[scores.index(min(scores))]

    return best_move
        
# a simple wrapper function as the display only gives one imput , BOARD
def play_min_maxN(BOARD):
    N=3
    return min_maxN(BOARD,N)