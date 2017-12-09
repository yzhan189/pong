
from State import *
from Board import *
from draw import *
import pygame,sys
from HardCode import *

# 0 human  / 1 AI     / 2 AIs
MODE = 1


redColour   = pygame.Color(255,0,0)
blackColour = pygame.Color(0,0,0)
whiteColour = pygame.Color(255,255,255)
greyColour = pygame.Color(200,200,200)
greenColour = pygame.Color(0,128,0)
yellowColour = pygame.Color(255,255,0)

pixel_size = 30

WINDOW_HEIGHT = HEIGHT*pixel_size
WINDOW_WIDTH = WIDTH*pixel_size

offset = 50
pygame.init()
pygame.mixer.init()
fpsClock = pygame.time.Clock()
playSurface = pygame.display.set_mode((WINDOW_WIDTH+200, WINDOW_HEIGHT+200))
playSurface.fill(whiteColour)
walls = []
if MODE!=2:
    for y in range(WINDOW_HEIGHT):
        walls.append([-3,y])
for x in range(WINDOW_WIDTH):
    walls.append([x,-3])
    walls.append([x,WINDOW_WIDTH+3])

def quit():
    pygame.quit()
    sys.exit()

def good_x_position(x):
    return int(x*WINDOW_WIDTH)

def good_y_position(y):
    return int(y*WINDOW_HEIGHT)


total_rewards = 0

key = None


board = Board()
state = State()
if MODE ==2:
    my_paddle = Left_Paddle()
else:
    my_paddle = None


paddle_y0 = 1
paddle_y1 = 1

my_paddle_y0 = 1
my_paddle_y1 = 1

ball_x = 0
ball_y = 0

Q = np.genfromtxt ('Q.csv', delimiter=",")

while True:
    # draw walls
    draw_surface(playSurface, blackColour, walls, 7, offset)

    # clear old paddle
    for yi in range(paddle_y0,paddle_y1):
        draw_surface(playSurface, whiteColour, [[WINDOW_WIDTH,yi]], 5, offset)

    # draw paddle
    paddle_y0 = good_y_position(state.paddle_y)
    paddle_y1 = paddle_y0 + good_y_position(state.paddle_height)
    for yi in range(paddle_y0,paddle_y1):
        draw_surface(playSurface, blackColour, [[WINDOW_WIDTH,yi]], 5, offset)

    # control the paddle
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            quit()
        elif event.type == pygame.KEYDOWN:
            if MODE == 0:
                # determine the event of keyBoard
                if event.key == pygame.K_UP or event.key == ord('w'):
                    state.move_paddle_up()
                if event.key == pygame.K_DOWN or event.key == ord('s'):
                    state.move_paddle_down()
            if event.key == pygame.K_ESCAPE:
                quit()

    # by AI
    if MODE == 1 or MODE ==2:
        s = state.discretize_get_index()
        a = np.argmax(Q[s])
        if a==0:
            state.move_paddle_up()
        elif a==2:
            state.move_paddle_down()

    # two AI
    if MODE==2:
        # clear paddle
        for yi in range(my_paddle_y0, my_paddle_y1):
            draw_surface(playSurface, whiteColour, [[0, yi]], 5, offset)

        # draw paddle
        my_paddle_y0 = good_y_position(my_paddle.y)
        my_paddle_y1 = my_paddle_y0 + good_y_position(my_paddle.paddle_height)
        for yi in range(my_paddle_y0, my_paddle_y1):
            draw_surface(playSurface, blackColour, [[0, yi]], 5, offset)

        # update paddle
        my_paddle.move(state.ball_y)


    # clear old ball but not wall
    draw_surface(playSurface, whiteColour, [[ball_x,ball_y]], 7, offset)

    # draw ball
    ball_y = good_y_position(state.ball_y)
    ball_x = good_x_position(state.ball_x)
    draw_surface(playSurface, redColour, [[ball_x,ball_y]], 7, offset)

    # clear score
    show_message(playSurface, "Total Rewards: " + str(total_rewards), whiteColour, 40, 10, 10)

    # ball movement and get rewards
    reward = state.move_ball_get_rewards(my_paddle)
    if reward == -1:
        total_rewards = 0
    else:
        total_rewards += reward

    # write score
    show_message(playSurface, "Total Rewards: " + str(total_rewards), blackColour, 40, 10, 10)


    pygame.display.flip()
    fpsClock.tick(20)



