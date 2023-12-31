import sys
import pygame
import math
import argparse

from ChessEngine import chess_visuals as chess_visuals
from Algorithm.MinMax import play_min_max
from Algorithm.Negamax import NegaMax, play_nega_max

from Algorithm.EvaluationFunctions import *

scrn = chess_visuals.scrn
board = chess_visuals.board

WHITE = chess_visuals.WHITE
GREY = chess_visuals.GREY
YELLOW = chess_visuals.YELLOW
BLUE = chess_visuals.BLUE
BLACK = chess_visuals.BLACK

def main_human_vs_human(BOARD):

    '''
    for human vs human game
    '''
    #make background black
    scrn.fill(BLACK)
    #name window
    pygame.display.set_caption('Chess')
    
    #variable to be used later
    index_moves = []

    status = True
    while (status):
        #update screen
        chess_visuals.update(scrn,BOARD)

        for event in pygame.event.get():
     
            # if event object type is QUIT
            # then quitting the pygame
            # and program both.
            if event.type == pygame.QUIT:
                status = False

            # if mouse clicked
            if event.type == pygame.MOUSEBUTTONDOWN:
                #remove previous highlights
                scrn.fill(BLACK)
                #get position of mouse
                pos = pygame.mouse.get_pos()

                #find which square was clicked and index of it
                square = (math.floor(pos[0]/100),math.floor(pos[1]/100))
                index = (7-square[1])*8+(square[0])
                
                # if we are moving a piece
                if index in index_moves: 
                    
                    move = moves[index_moves.index(index)]
                    
                    BOARD.push(move)

                    #reset index and moves
                    index=None
                    index_moves = []
                    
                    
                # show possible moves
                else:
                    #check the square that is clicked
                    piece = BOARD.piece_at(index)
                    #if empty pass
                    if piece == None:
                        
                        pass
                    else:
                        
                        #figure out what moves this piece can make
                        all_moves = list(BOARD.legal_moves)
                        moves = []
                        for m in all_moves:
                            if m.from_square == index:
                                
                                moves.append(m)

                                t = m.to_square

                                TX1 = 100*(t%8)
                                TY1 = 100*(7-t//8)

                                
                                #highlight squares it can move to
                                pygame.draw.rect(scrn,BLUE,pygame.Rect(TX1,TY1,100,100),5)
                        
                        index_moves = [a.to_square for a in moves]
     
    # deactivates the pygame library
        if BOARD.outcome() != None:
            print(BOARD.outcome())
            status = False
            print(BOARD)
    pygame.quit()

def main_one_agent(BOARD,agent,agent_color):
    
    '''
    for agent vs human game
    color is True = White agent
    color is False = Black agent
    '''
    #make background black
    scrn.fill(BLACK)
    #name window
    pygame.display.set_caption('Chess')
    
    #variable to be used later
    index_moves = []

    status = True
    while (status):
        #update screen
        chess_visuals.update(scrn,BOARD)
        
     
        if BOARD.turn == agent_color:
            BOARD.push(agent.select_move(BOARD))
            scrn.fill(BLACK)

        else:

            for event in pygame.event.get():
         
                # if event object type is QUIT
                # then quitting the pygame
                # and program both.
                if event.type == pygame.QUIT:
                    status = False

                # if mouse clicked
                if event.type == pygame.MOUSEBUTTONDOWN:
                    #reset previous screen from clicks
                    scrn.fill(BLACK)
                    #get position of mouse
                    pos = pygame.mouse.get_pos()

                    #find which square was clicked and index of it
                    square = (math.floor(pos[0]/100),math.floor(pos[1]/100))
                    index = (7-square[1])*8+(square[0])
                    
                    # if we have already highlighted moves and are making a move
                    if index in index_moves: 
                        
                        move = moves[index_moves.index(index)]
                        #print(BOARD)
                        #print(move)
                        BOARD.push(move)
                        index=None
                        index_moves = []
                        
                    # show possible moves
                    else:
                        
                        piece = BOARD.piece_at(index)
                        
                        if piece == None:
                            
                            pass
                        else:

                            all_moves = list(BOARD.legal_moves)
                            moves = []
                            for m in all_moves:
                                if m.from_square == index:
                                    
                                    moves.append(m)

                                    t = m.to_square

                                    TX1 = 100*(t%8)
                                    TY1 = 100*(7-t//8)

                                    
                                    pygame.draw.rect(scrn,BLUE,pygame.Rect(TX1,TY1,100,100),5)
                            #print(moves)
                            index_moves = [a.to_square for a in moves]
     
    # deactivates the pygame library
        if BOARD.outcome() != None:
            print(BOARD.outcome())
            status = False
            print(BOARD)
    
    pygame.quit()
    

def main_two_agent(BOARD,agent1,agent_color1,agent2):
    '''
    for agent vs agent game
    
    '''
    #make background black
    scrn.fill(BLACK)
    #name window
    pygame.display.set_caption('Chess')
    
    #variable to be used later

    status = True
    while (status):
        #update screen
        chess_visuals.update(scrn,BOARD)
        
        if BOARD.turn == agent_color1:
            BOARD.push(agent1.select_move(BOARD))
        else:
            BOARD.push(agent2.select_move(BOARD))

        scrn.fill(BLACK)
            
        for event in pygame.event.get():
     
            # if event object type is QUIT
            # then quitting the pygame
            # and program both.
            if event.type == pygame.QUIT:
                status = False
     
    # deactivates the pygame library
        if BOARD.outcome() != None:
            print(BOARD.outcome())
            status = False
            print(BOARD)
    pygame.quit()



# agent_mapping = {
#     "min_max": play_min_max,
#     "nega_max": play_nega_max
# }

# def evaluate_functions_on_board(BOARD, evaluation_functions):
#     for name, eval_func in evaluation_functions.items():
#         negamax = NegaMax(search_depth=3, evaluation_function=eval_func, player='white')
#         best_move = negamax.find_best_move(copy.deepcopy(BOARD))
#         score = negamax.run_negamax(BOARD, 3, float('-inf'), float('inf'))
#         print(f"Evaluation Function: {name}, Best Move: {best_move}, Score: {score}")

def main():
    parser = argparse.ArgumentParser(description="Chess game modes and AI options")
    parser.add_argument("mode", nargs='?', default='two_agents', help="Game mode: 'human', 'one_agent', or 'two_agents'")
    parser.add_argument("--agent1", help="First AI agent: 'min_max' or 'nega_max'", default="nega_max")
    parser.add_argument("--agent2", help="Second AI agent: 'min_max' or 'nega_max' (only for two_agents mode)", default="nega_max")
    parser.add_argument("--evaluator1", help="Evaluator for the first AI agent", default="relative_basic_piece_squares_and_material")
    parser.add_argument("--evaluator2", help="Evaluator for the second AI agent (only for two_agents mode)", default="relative_simple_material")
    parser.add_argument("--agent_color", help="AI agent color: 'white' or 'black' (choose color for your opponent agent or first agent in two_agent mode)", default="white")
    # parser.add_argument("--eval_func", help="Evaluation function for negamax algorithm", default="simple_material")
    args = parser.parse_args()

    pygame.init()
    board = chess_visuals.board
    agent_mapping = {
        "min_max": play_min_max,
        "nega_max": play_nega_max
    }

    evaluator_mapping = {
        "simple_material": simple_material_evaluator,
        "stockfish": stockfish_evaluator,
        "basic_piece_squares": basic_piece_squares_and_material_evaluator,
        "tapered_piece_squares": tapered_piece_squares_evaluator,
        "king_safety": king_safety_evaluator,
        "simple_piece_count": simple_piece_count_evaluator,
        "pieces_attacked": pieces_attacked_evaluator,
        "relative_simple_material": relative_simple_material_evaluator,
        "relative_basic_piece_squares": relative_basic_piece_squares_and_material_evaluator,
        "relative_tapered_piece_squares": relative_tapered_piece_squares_evaluator,
        "relative_king_safety": relative_king_safety_evaluator,
        "relative_simple_piece_count": relative_simple_piece_count_evaluator,
        "relative_pieces_attacked": relative_pieces_attacked_evaluator,
        "relative_polynomial_regression": relative_polynomial_regression_evaluator,
        "relative_random_forest_classifier": relative_random_forest_classifier_evaluator,
        "relative_random_forest_regression": relative_random_forest_regression_evaluator,
        "polynomial_regression": polynomial_regression_evaluator,
        "random_forest_classifier": random_forest_classifier_evaluator,
        "random_forest_regression": random_forest_regression_evaluator,
        "basic_piece_squares_and_material": basic_piece_squares_and_material_evaluator,
        "relative_basic_piece_squares_and_material": relative_basic_piece_squares_and_material_evaluator
    }

    evaluator1 = evaluator_mapping[args.evaluator1]
    evaluator2 = evaluator_mapping[args.evaluator2]

    if args.mode == 'human':
        main_human_vs_human(board)
    elif args.mode == 'one_agent':
        player_color_bool = args.agent_color.lower() == 'white'
        main_one_agent(board, agent_mapping[args.agent1](evaluator1), player_color_bool)
    elif args.mode == 'two_agents':
        player_color_bool = args.agent_color.lower() == 'white'
        main_two_agent(board, agent_mapping[args.agent1](evaluator1), player_color_bool, agent_mapping[args.agent2](evaluator2))
    else:
        print("Invalid game mode. Choose from 'human', 'one_agent', or 'two_agents'.")
if __name__ == "__main__":
    main()
