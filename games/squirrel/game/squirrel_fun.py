# Squirrel Eat Squirrel (a 2D Katamari Damacy clone)
# By Al Sweigart al@inventwithpython.com
# http://inventwithpython.com/pygame
# Released under a "Simplified BSD" license


import random, sys, time, math, pygame
from pygame.locals import *

sys.path.append("game/")

WINWIDTH = 640 # width of the program's window, in pixels
WINHEIGHT = 480 # height in pixels
HALF_WINWIDTH = int(WINWIDTH / 2)
HALF_WINHEIGHT = int(WINHEIGHT / 2)

GRASSCOLOR = (24, 255, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)

CAMERASLACK = 90     # how far from the center the squirrel moves before moving the camera
MOVERATE = 9         # how fast the player moves
BOUNCERATE = 6       # how fast the player bounces (large is slower)
BOUNCEHEIGHT = 30    # how high the player bounces
STARTSIZE = 25       # how big the player starts off
WINSIZE = 300        # how big the player needs to be to win
INVULNTIME = 2       # how long the player is invulnerable after being hit in seconds
GAMEOVERTIME = 4     # how long the "game over" text stays on the screen in seconds
MAXHEALTH = 1        # how much health the player starts with

NUMGRASS = 80        # number of grass objects in the active area
NUMSQUIRRELS = 30    # number of squirrels in the active area
SQUIRRELMINSPEED = 3 # slowest squirrel speed
SQUIRRELMAXSPEED = 7 # fastest squirrel speed
DIRCHANGEFREQ = 2    # % chance of direction change per frame
LEFT = 'left'
RIGHT = 'right'

#rewards
live_reward=0.1
lose_reward=-1
win_reward=1

"""
This program has three data structures to represent the player, enemy squirrels, and grass background objects. The data structures are dictionaries with the following keys:

Keys used by all three data structures:
    'x' - the left edge coordinate of the object in the game world (not a pixel coordinate on the screen)
    'y' - the top edge coordinate of the object in the game world (not a pixel coordinate on the screen)
    'rect' - the pygame.Rect object representing where on the screen the object is located.
Player data structure keys:
    'surface' - the pygame.Surface object that stores the image of the squirrel which will be drawn to the screen.
    'facing' - either set to LEFT or RIGHT, stores which direction the player is facing.
    'size' - the width and height of the player in pixels. (The width & height are always the same.)
    'bounce' - represents at what point in a bounce the player is in. 0 means standing (no bounce), up to BOUNCERATE (the completion of the bounce)
    'health' - an integer showing how many more times the player can be hit by a larger squirrel before dying.
Enemy Squirrel data structure keys:
    'surface' - the pygame.Surface object that stores the image of the squirrel which will be drawn to the screen.
    'movex' - how many pixels per frame the squirrel moves horizontally. A negative integer is moving to the left, a positive to the right.
    'movey' - how many pixels per frame the squirrel moves vertically. A negative integer is moving up, a positive moving down.
    'width' - the width of the squirrel's image, in pixels
    'height' - the height of the squirrel's image, in pixels
    'bounce' - represents at what point in a bounce the player is in. 0 means standing (no bounce), up to BOUNCERATE (the completion of the bounce)
    'bouncerate' - how quickly the squirrel bounces. A lower number means a quicker bounce.
    'bounceheight' - how high (in pixels) the squirrel bounces
Grass data structure keys:
    'grassImage' - an integer that refers to the index of the pygame.Surface object in GRASSIMAGES used for this grass object
"""
class GameState:
    def __init__(self):
        global DISPLAYSURF, BASICFONT, L_SQUIR_IMG, R_SQUIR_IMG, GRASSIMAGES
        pygame.init()
        pygame.display.set_icon(pygame.image.load('assets/gameicon.png'))
        DISPLAYSURF = pygame.display.set_mode((WINWIDTH, WINHEIGHT))
        pygame.display.set_caption('Squirrel Eat Squirrel')
        BASICFONT = pygame.font.Font('freesansbold.ttf', 32)
        # load the image files
        L_SQUIR_IMG = pygame.image.load('assets/squirrel.png')
        R_SQUIR_IMG = pygame.transform.flip(L_SQUIR_IMG, True, False)
        GRASSIMAGES = []
        for i in range(1, 5):
            GRASSIMAGES.append(pygame.image.load('assets/grass%s.png' % i))

        self.reinit()


    def reinit(self):
        # set up variables for the start of a new game
        # camerax and cameray are the top left of where the camera view is
        self.camerax = 0
        self.cameray = 0
        self.score=0
        self.invulnerableMode = False  # if the player is invulnerable
        self.invulnerableStartTime = 0  # time the player became invulnerable
        self.gameOverMode = False  # if the player has lost
        self.winMode = False  # if the player has won
        self.grassObjs = []  # stores all the grass objects in the game
        self.squirrelObjs = []  # stores all the non-player squirrel objects
        # stores the player object:
        self.playerObj = {'surface': pygame.transform.scale(L_SQUIR_IMG, (STARTSIZE, STARTSIZE)),
                     'facing': LEFT,
                     'size': STARTSIZE,
                     'x': HALF_WINWIDTH,
                     'y': HALF_WINHEIGHT,
                     'bounce': 0,
                     'health': MAXHEALTH}

        #start off with some random grass images on the screen
        for i in range(10):
            self.grassObjs.append(makeNewGrass(self.camerax, self.cameray))
            self.grassObjs[i]['x'] = random.randint(0, WINWIDTH)
            self.grassObjs[i]['y'] = random.randint(0, WINHEIGHT)


    def frame_step(self, input):
        self.moveLeft = False
        self.moveRight = False
        self.moveUp = False
        self.moveDown = False

        reward = live_reward
        terminal = False

        # Check if we should turn off invulnerability
        if self.invulnerableMode and time.time() - self.invulnerableStartTime > INVULNTIME:
            self.invulnerableMode = False

        # move all the squirrels
        for sObj in self.squirrelObjs:
            # move the squirrel, and adjust for their bounce
            sObj['x'] += sObj['movex']
            sObj['y'] += sObj['movey']
            sObj['bounce'] += 1
            if sObj['bounce'] > sObj['bouncerate']:
                sObj['bounce'] = 0  # reset bounce amount

            # random chance they change direction
            if random.randint(0, 99) < DIRCHANGEFREQ:
                sObj['movex'] = getRandomVelocity()
                sObj['movey'] = getRandomVelocity()
                if sObj['movex'] > 0:  # faces right
                    sObj['surface'] = pygame.transform.scale(R_SQUIR_IMG, (sObj['width'], sObj['height']))
                else:  # faces left
                    sObj['surface'] = pygame.transform.scale(L_SQUIR_IMG, (sObj['width'], sObj['height']))

        # go through all the objects and see if any need to be deleted.
        for i in range(len(self.grassObjs) - 1, -1, -1):
            if isOutsideActiveArea(self.camerax, self.cameray, self.grassObjs[i]):
                del self.grassObjs[i]
        for i in range(len(self.squirrelObjs) - 1, -1, -1):
            if isOutsideActiveArea(self.camerax, self.cameray, self.squirrelObjs[i]):
                del self.squirrelObjs[i]

        # add more grass & squirrels if we don't have enough.
        while len(self.grassObjs) < NUMGRASS:
            self.grassObjs.append(makeNewGrass(self.camerax, self.cameray))
        while len(self.squirrelObjs) < NUMSQUIRRELS:
            self.squirrelObjs.append(makeNewSquirrel(self.camerax, self.cameray))

        # adjust camerax and cameray if beyond the "camera slack"
        playerCenterx = self.playerObj['x'] + int(self.playerObj['size'] / 2)
        playerCentery = self.playerObj['y'] + int(self.playerObj['size'] / 2)
        if (self.camerax + HALF_WINWIDTH) - playerCenterx > CAMERASLACK:
            self.camerax = playerCenterx + CAMERASLACK - HALF_WINWIDTH
        elif playerCenterx - (self.camerax + HALF_WINWIDTH) > CAMERASLACK:
            self.camerax = playerCenterx - CAMERASLACK - HALF_WINWIDTH
        if (self.cameray + HALF_WINHEIGHT) - playerCentery > CAMERASLACK:
            self.cameray = playerCentery + CAMERASLACK - HALF_WINHEIGHT
        elif playerCentery - (self.cameray + HALF_WINHEIGHT) > CAMERASLACK:
            self.cameray = playerCentery - CAMERASLACK - HALF_WINHEIGHT

        # draw the green background
        DISPLAYSURF.fill(GRASSCOLOR)

        # draw all the grass objects on the screen
        for gObj in self.grassObjs:
            gRect = pygame.Rect((gObj['x'] - self.camerax,
                                 gObj['y'] - self.cameray,
                                 gObj['width'],
                                 gObj['height']))
            DISPLAYSURF.blit(GRASSIMAGES[gObj['grassImage']], gRect)

        # draw the other squirrels
        for sObj in self.squirrelObjs:
            sObj['rect'] = pygame.Rect((sObj['x'] - self.camerax,
                                          sObj['y'] - self.cameray - getBounceAmount(sObj['bounce'], sObj['bouncerate'],
                                                                                sObj['bounceheight']),
                                          sObj['width'],
                                        sObj['height']))
            DISPLAYSURF.blit(sObj['surface'], sObj['rect'])

        # draw the player squirrel
        flashIsOn = round(time.time(), 1) * 10 % 2 == 1
        if not self.gameOverMode and not (self.invulnerableMode and flashIsOn):
            self.playerObj['rect'] = pygame.Rect((self.playerObj['x'] - self.camerax,
                                                  self.playerObj['y'] - self.cameray - getBounceAmount(self.playerObj['bounce'],
                                                                                            BOUNCERATE, BOUNCEHEIGHT),
                                                  self.playerObj['size'],
                                                  self.playerObj['size']))
            DISPLAYSURF.blit(self.playerObj['surface'], self.playerObj['rect'])

        if (input[1] == 1): #K_UP
            self.moveDown = False
            self.moveUp = True

        elif(input[2] == 1): #K_DOWN
            self.moveUp = False
            self.moveDown = True

        elif (input[3] == 1): #K_LEFT
            self.moveRight = False
            self.moveLeft = True
            if self.playerObj['facing'] != LEFT:  # change player image
                self.playerObj['surface'] = pygame.transform.scale(L_SQUIR_IMG,(self.playerObj['size'], self.playerObj['size']))
            self.playerObj['facing'] = LEFT

        elif (input[4] == 1): #K_RIGHT
            self.moveLeft = False
            self.moveRight = True
            if self.playerObj['facing'] != RIGHT:  # change player image
                self.playerObj['surface'] = pygame.transform.scale(R_SQUIR_IMG,
                                                                          (self.playerObj['size'], self.playerObj['size']))
                self.playerObj['facing'] = RIGHT

        if not self.gameOverMode:
            # actually move the player
            if self.moveLeft:
                self.playerObj['x'] -= MOVERATE
            if self.moveRight:
                self.playerObj['x'] += MOVERATE
            if self.moveUp:
                self.playerObj['y'] -= MOVERATE
            if self.moveDown:
                self.playerObj['y'] += MOVERATE

            if (self.moveLeft or self.moveRight or self.moveUp or self.moveDown) or self.playerObj['bounce'] != 0:
                self.playerObj['bounce'] += 1

            if self.playerObj['bounce'] > BOUNCERATE:
                self.playerObj['bounce'] = 0  # reset bounce amount

            # check if the player has collided with any squirrels
            for i in range(len(self.squirrelObjs) - 1, -1, -1):
                sqObj = self.squirrelObjs[i]
                if 'rect' in sqObj and self.playerObj['rect'].colliderect(sqObj['rect']):
                    # a player/squirrel collision has occurred

                    if sqObj['width'] * sqObj['height'] <= self.playerObj['size'] ** 2:
                        # player is larger and eats the squirrel
                        self.playerObj['size'] += int((sqObj['width'] * sqObj['height']) ** 0.2) + 1
                        del self.squirrelObjs[i]
                        reward = win_reward
                        self.score+=10

                        if self.playerObj['facing'] == LEFT:
                            self.playerObj['surface'] = pygame.transform.scale(L_SQUIR_IMG,
                                                                              (self.playerObj['size'], self.playerObj['size']))
                        if self.playerObj['facing'] == RIGHT:
                            self.playerObj['surface'] = pygame.transform.scale(R_SQUIR_IMG,
                                                                              (self.playerObj['size'], self.playerObj['size']))
                        if self.playerObj['size'] > WINSIZE:
                            reward=win_reward
                            #self.winMode = True  # turn on "win mode"
                            self.drawStatus()
                            pygame.display.update()
                            image_data = pygame.surfarray.array3d(pygame.display.get_surface())
                            terminal = True
                            temp_score = self.score
                            self.reinit()
                            return image_data, reward, terminal, temp_score


                    elif not self.invulnerableMode:
                    # player is smaller and takes damage
                        self.invulnerableMode = True
                        self.invulnerableStartTime = time.time()
                        self.playerObj['health'] -= 1
                        terminal = True

                    if self.playerObj['health'] == 0:
                            self.drawStatus()
                            pygame.display.update()
                            image_data = pygame.surfarray.array3d(pygame.display.get_surface())
                            terminal = True
                            reward = lose_reward
                            #raw_input("Press Enter to continue...")
                            temp_score = self.score
                            self.reinit()
                            return image_data, reward, terminal, temp_score


        else:
            return  # end the current game


        self.drawStatus()
        pygame.display.update()


        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        return image_data, reward, terminal, self.score



    def drawStatus(self):
        # create the surfaces to hold game text
        gameOverSurf = BASICFONT.render('Game Over', True, WHITE)
        gameOverRect = gameOverSurf.get_rect()
        gameOverRect.center = (HALF_WINWIDTH, HALF_WINHEIGHT)

        winSurf = BASICFONT.render('You have achieved OMEGA SQUIRREL!', True, WHITE)
        winRect = winSurf.get_rect()
        winRect.center = (HALF_WINWIDTH, HALF_WINHEIGHT)

        winSurf2 = BASICFONT.render('(Press "r" to restart.)', True, WHITE)
        winRect2 = winSurf2.get_rect()
        winRect2.center = (HALF_WINWIDTH, HALF_WINHEIGHT + 30)





def terminate():
    pygame.quit()
    sys.exit()


def getBounceAmount(currentBounce, bounceRate, bounceHeight):
    # Returns the number of pixels to offset based on the bounce.
    # Larger bounceRate means a slower bounce.
    # Larger bounceHeight means a higher bounce.
    # currentBounce will always be less than bounceRate
    return int(math.sin( (math.pi / float(bounceRate)) * currentBounce ) * bounceHeight)

def getRandomVelocity():
    speed = random.randint(SQUIRRELMINSPEED, SQUIRRELMAXSPEED)
    if random.randint(0, 1) == 0:
        return speed
    else:
        return -speed


def getRandomOffCameraPos(camerax, cameray, objWidth, objHeight):
    # create a Rect of the camera view
    cameraRect = pygame.Rect(camerax, cameray, WINWIDTH, WINHEIGHT)
    while True:
        x = random.randint(camerax - WINWIDTH, camerax + (2 * WINWIDTH))
        y = random.randint(cameray - WINHEIGHT, cameray + (2 * WINHEIGHT))
        # create a Rect object with the random coordinates and use colliderect()
        # to make sure the right edge isn't in the camera view.
        objRect = pygame.Rect(x, y, objWidth, objHeight)
        if not objRect.colliderect(cameraRect):
            return x, y


def makeNewSquirrel(camerax, cameray):
    sq = {}
    generalSize = random.randint(5, 25)
    multiplier = random.randint(1, 3)
    sq['width']  = (generalSize + random.randint(0, 10)) * multiplier
    sq['height'] = (generalSize + random.randint(0, 10)) * multiplier
    sq['x'], sq['y'] = getRandomOffCameraPos(camerax, cameray, sq['width'], sq['height'])
    sq['movex'] = getRandomVelocity()
    sq['movey'] = getRandomVelocity()
    if sq['movex'] < 0: # squirrel is facing left
        sq['surface'] = pygame.transform.scale(L_SQUIR_IMG, (sq['width'], sq['height']))
    else: # squirrel is facing right
        sq['surface'] = pygame.transform.scale(R_SQUIR_IMG, (sq['width'], sq['height']))
    sq['bounce'] = 0
    sq['bouncerate'] = random.randint(10, 18)
    sq['bounceheight'] = random.randint(10, 50)
    return sq


def makeNewGrass(camerax, cameray):
    gr = {}
    gr['grassImage'] = random.randint(0, len(GRASSIMAGES) - 1)
    gr['width']  = GRASSIMAGES[0].get_width()
    gr['height'] = GRASSIMAGES[0].get_height()
    gr['x'], gr['y'] = getRandomOffCameraPos(camerax, cameray, gr['width'], gr['height'])
    gr['rect'] = pygame.Rect( (gr['x'], gr['y'], gr['width'], gr['height']) )
    return gr


def isOutsideActiveArea(camerax, cameray, obj):
    # Return False if camerax and cameray are more than
    # a half-window length beyond the edge of the window.
    boundsLeftEdge = camerax - WINWIDTH
    boundsTopEdge = cameray - WINHEIGHT
    boundsRect = pygame.Rect(boundsLeftEdge, boundsTopEdge, WINWIDTH * 3, WINHEIGHT * 3)
    objRect = pygame.Rect(obj['x'], obj['y'], obj['width'], obj['height'])
    return not boundsRect.colliderect(objRect)



