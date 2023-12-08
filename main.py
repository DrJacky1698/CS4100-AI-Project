import sys
import pygame
import math

from ChessEngine import chess_visuals as chess_visuals
from Algorithm.MinMax import play_min_maxN
from Algorithm.Negamax import play_nega_max

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
        
     
        if BOARD.turn==agent_color:
            BOARD.push(agent(BOARD))
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
        
        if BOARD.turn==agent_color1:
            BOARD.push(agent1(BOARD))

        else:
            BOARD.push(agent2(BOARD))

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

# def main(game_mode, board, agent=None, agent_color=None, second_agent=None):
#     '''
#     Initialize the game based on the game mode.
#     :param game_mode: A string indicating the game mode ('human', 'one_agent', 'two_agents')
#     :param board: The chess board to be used
#     :param agent: The agent function for one player or the first agent in two agents mode
#     :param agent_color: The color for the agent ('white' or 'black')
#     :param second_agent: The second agent function for two agents mode
#     '''
#     if game_mode == 'human':
#         main_human_vs_human(board)
#     elif game_mode == 'one_agent' and agent is not None:
#         main_one_agent(board, agent, agent_color)
#     elif game_mode == 'two_agents' and agent is not None and second_agent is not None:
#         main_two_agent(board, agent, agent_color, second_agent)

# if __name__ == "__main__":
#     # Example usage:
#     # python main.py human
#     # python main.py one_agent white
#     # python main.py two_agents

#     game_mode = sys.argv[1] if len(sys.argv) > 1 else 'human'
#     agent_color = sys.argv[2] if len(sys.argv) > 2 else 'white'

#     # Start the game
#     if game_mode == 'human':
#         main(game_mode, board)
#     elif game_mode == 'one_agent':
#         main(game_mode, board, agent=example_agent, agent_color=agent_color)
#     elif game_mode == 'two_agents':
#         main(game_mode, board, agent=example_agent, agent_color=agent_color, second_agent=example_agent)
#     else:
#         print("Unknown game mode.")

# main_two_agent(board, play_min_maxN, False, play_min_maxN)
main_two_agent(board, play_nega_max, False, play_nega_max)
#Hi simon