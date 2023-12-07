# importing required librarys
import pygame
import chess

#initialise display
X = 800
Y = 800
scrn = pygame.display.set_mode((X, Y))
pygame.init()

#basic colours
WHITE = (255, 255, 255)
GREY = (128, 128, 128)
YELLOW = (204, 204, 0)
BLUE = (50, 255, 255)
BLACK = (0, 0, 0)

#initialise chess board
board = chess.Board()

#load piece images
pieces = {'p': pygame.image.load('Images/b-pawn.png').convert(),
          'n': pygame.image.load('Images/b-knight.png').convert(),
          'b': pygame.image.load('Images/b-bishop.png').convert(),
          'r': pygame.image.load('Images/b-rook.png').convert(),
          'q': pygame.image.load('Images/b-queen.png').convert(),
          'k': pygame.image.load('Images/b-king.png').convert(),
          'P': pygame.image.load('Images/w-pawn.png').convert(),
          'N': pygame.image.load('Images/w-knight.png').convert(),
          'B': pygame.image.load('Images/w-bishop.png').convert(),
          'R': pygame.image.load('Images/w-rook.png').convert(),
          'Q': pygame.image.load('Images/w-queen.png').convert(),
          'K': pygame.image.load('Images/w-king.png').convert(),
          
          }

def update(scrn,board):
    '''
    updates the screen basis the board class
    '''
    
    for i in range(64):
        piece = board.piece_at(i)
        if piece == None:
            pass
        else:
            scrn.blit(pieces[str(piece)],((i%8)*100,700-(i//8)*100))
    
    for i in range(7):
        i=i+1
        pygame.draw.line(scrn,WHITE,(0,i*100),(800,i*100))
        pygame.draw.line(scrn,WHITE,(i*100,0),(i*100,800))

    pygame.display.flip()