# Wormy (a Nibbles clone)
# By Al Sweigart al@inventwithpython.com
# http://inventwithpython.com/pygame
# Released under a "Simplified BSD" license

#KRT 14/06/2012 modified Start Screen and Game Over screen to cope with mouse events
#KRT 14/06/2012 Added a non-busy wait to Game Over screen to reduce processor loading from near 100%
import random, pygame, sys
from pygame.locals import *

FPS = 15
WINDOWWIDTH = 240
WINDOWHEIGHT = 240
CELLSIZE = 20
assert WINDOWWIDTH % CELLSIZE == 0, "Window width must be a multiple of cell size."
assert WINDOWHEIGHT % CELLSIZE == 0, "Window height must be a multiple of cell size."
CELLWIDTH = int(WINDOWWIDTH / CELLSIZE)
CELLHEIGHT = int(WINDOWHEIGHT / CELLSIZE)

#             R    G    B
WHITE     = (255, 255, 255)
BLACK     = (  0,   0,   0)
RED       = (255,   0,   0)
GREEN     = (  0, 255,   0)
DARKGREEN = (  0, 155,   0)
DARKGRAY  = ( 40,  40,  40)
BGCOLOR = BLACK

UP = 'up'
DOWN = 'down'
LEFT = 'left'
RIGHT = 'right'

#rewards
lose_reward=-1
win_reward=1
live_reward=-0.01

HEAD = 0 # syntactic sugar: index of the worm's head

class GameState:
    def __init__(self):
        global DISPLAYSURF, BASICFONT
        pygame.init()
        DISPLAYSURF = pygame.display.set_mode((WINDOWWIDTH, WINDOWHEIGHT))
        BASICFONT = pygame.font.Font('freesansbold.ttf', 18)
        pygame.display.set_caption('Snake')

        self.reinit()

        pygame.display.update()

    def reinit(self):
        self.HEAD = 0 # syntactic sugar: index of the worm's head
        # Set a random start point.
        self.startx = random.randint(5, CELLWIDTH - 6)
        self.starty = random.randint(5, CELLHEIGHT - 6)

        self.wormCoords = [{'x': self.startx,     'y': self.starty},
                      {'x': self.startx - 1, 'y': self.starty},
                      {'x': self.startx - 2, 'y': self.starty}]
        self.direction = RIGHT
        # Start the apple in a random place.
        self.apple = self.getRandomLocation()

        pygame.display.update()

    def frame_step(self, input_vect):
        reward = live_reward
        terminal = False

        if input_vect[1]  == 1 and self.direction != RIGHT:  # Key left
            self.direction = LEFT

        elif input_vect[2] == 1 and self.direction != LEFT:  # Key right
            self.direction = RIGHT

        elif input_vect[3] == 1 and self.direction != DOWN:  # Key up
            self.direction = UP

        elif input_vect[4] == 1 and self.direction != UP:  # Key down
            self.direction = DOWN

        # check if the worm has hit itself or the edge
        if self.wormCoords[self.HEAD]['x'] == -1 or self.wormCoords[self.HEAD]['x'] == CELLWIDTH or self.wormCoords[self.HEAD]['y'] == -1 or self.wormCoords[self.HEAD]['y'] == CELLHEIGHT:
            terminal = True
        for wormBody in self.wormCoords[1:]:
            if wormBody['x'] == self.wormCoords[self.HEAD]['x'] and wormBody['y'] == self.wormCoords[self.HEAD]['y']:
                terminal=True

        if(terminal):
            image_data = pygame.surfarray.array3d(pygame.display.get_surface())
            reward = lose_reward
            temp_score = len(self.wormCoords) - 3
            self.reinit()
            return image_data, reward, terminal, temp_score  # can't fit a new piece on the self.board, so game over

        # check if worm has eaten an apply
        if self.wormCoords[self.HEAD]['x'] == self.apple['x'] and self.wormCoords[self.HEAD]['y'] == self.apple['y']:
            # don't remove worm's tail segment
            self.apple = self.getRandomLocation() # set a new apple somewhere
            reward=win_reward


        else:
            del self.wormCoords[-1] # remove worm's tail segment

        # move the worm by adding a segment in the direction it is moving
        if self.direction == UP:
            newHead = {'x': self.wormCoords[self.HEAD]['x'], 'y': self.wormCoords[self.HEAD]['y'] - 1}
        elif self.direction == DOWN:
            newHead = {'x': self.wormCoords[self.HEAD]['x'], 'y': self.wormCoords[self.HEAD]['y'] + 1}
        elif self.direction == LEFT:
            newHead = {'x': self.wormCoords[self.HEAD]['x'] - 1, 'y': self.wormCoords[self.HEAD]['y']}
        elif self.direction == RIGHT:
            newHead = {'x': self.wormCoords[self.HEAD]['x'] + 1, 'y': self.wormCoords[self.HEAD]['y']}

        self.wormCoords.insert(0, newHead)
        DISPLAYSURF.fill(BGCOLOR)
        self.drawGrid()
        self.drawWorm(self.wormCoords)
        self.drawApple(self.apple)
        score=len(self.wormCoords) - 3
        self.drawScore(score)

        pygame.display.update()


        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        return image_data, reward, terminal, score

    def drawPressKeyMsg(self):
        pressKeySurf = BASICFONT.render('Press a key to play.', True, DARKGRAY)
        pressKeyRect = pressKeySurf.get_rect()
        pressKeyRect.topleft = (WINDOWWIDTH - 200, WINDOWHEIGHT - 30)
        DISPLAYSURF.blit(pressKeySurf, pressKeyRect)

    def getRandomLocation(self):
        return {'x': random.randint(0, CELLWIDTH - 1), 'y': random.randint(0, CELLHEIGHT - 1)}

    def drawScore(self,score):
        scoreSurf = BASICFONT.render('Score: %s' % (score), True, WHITE)
        scoreRect = scoreSurf.get_rect()
        scoreRect.topleft = (WINDOWWIDTH - 120, 10)
        DISPLAYSURF.blit(scoreSurf, scoreRect)


    def drawWorm(self,wormCoords):
        for coord in self.wormCoords:
            x = coord['x'] * CELLSIZE
            y = coord['y'] * CELLSIZE
            wormSegmentRect = pygame.Rect(x, y, CELLSIZE, CELLSIZE)
            pygame.draw.rect(DISPLAYSURF, DARKGREEN, wormSegmentRect)
            wormInnerSegmentRect = pygame.Rect(x + 4, y + 4, CELLSIZE - 8, CELLSIZE - 8)
            pygame.draw.rect(DISPLAYSURF, GREEN, wormInnerSegmentRect)

    def drawApple(self,coord):
        x = coord['x'] * CELLSIZE
        y = coord['y'] * CELLSIZE
        appleRect = pygame.Rect(x, y, CELLSIZE, CELLSIZE)
        pygame.draw.rect(DISPLAYSURF, RED, appleRect)

    def drawGrid(self):
        for x in range(0, WINDOWWIDTH, CELLSIZE): # draw vertical lines
            pygame.draw.line(DISPLAYSURF, DARKGRAY, (x, 0), (x, WINDOWHEIGHT))
        for y in range(0, WINDOWHEIGHT, CELLSIZE): # draw horizontal lines
            pygame.draw.line(DISPLAYSURF, DARKGRAY, (0, y), (WINDOWWIDTH, y))

