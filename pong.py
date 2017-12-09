import curses
from curses import KEY_UP, KEY_DOWN
from State import *
from Board import *
import pygame

pixel_size = 50

WINDOW_HEIGHT = HEIGHT*pixel_size
WINDOW_WIDTH = WIDTH*pixel_size


def good_x_position(x):
    return int(x*WINDOW_WIDTH)

def good_y_position(y):
    return int(y*WINDOW_HEIGHT)

# set up window
curses.initscr()
win = curses.newwin(WINDOW_HEIGHT, WINDOW_WIDTH, 0, 0)
win.border(curses.ACS_VLINE,' ',curses.ACS_HLINE,curses.ACS_HLINE,curses.ACS_ULCORNER,' ',curses.ACS_LLCORNER,' ')
win.keypad(True)
curses.curs_set(0)

total_rewards = 0

key = None


board = Board()
state = State()

paddle_y0 = 1
paddle_y1 = 1

ball_x = 1
ball_y = 1

while key!=27:
    win.timeout(30)
    # write score
    win.addstr(0, WINDOW_WIDTH//2 - 10, 'Total Rewards: ' + str(total_rewards))

    # clear old paddle
    for yi in range(paddle_y0,paddle_y1):
        win.addstr(yi, WINDOW_WIDTH-1, ' ' )

    # draw paddle
    paddle_y0 = good_y_position(state.paddle_y)
    paddle_y1 = paddle_y0 + good_y_position(state.paddle_height)
    for yi in range(paddle_y0,paddle_y1):
        win.addstr(yi, WINDOW_WIDTH-1, '|' )

    # control the paddle
    key = win.getch()
    if key == KEY_UP:
        state.move_paddle_up()
    if key == KEY_DOWN:
        state.move_paddle_down()

    # clear old ball but not wall
    if ball_y != 0 and ball_x != 0 and ball_y != WINDOW_HEIGHT:
        win.addstr(ball_y, ball_x, ' ')

    # draw ball
    ball_y = good_y_position(state.ball_y)
    ball_x = good_x_position(state.ball_x)
    if ball_y != 0 and ball_x != 0 and ball_y != WINDOW_HEIGHT:

        win.addstr(ball_y, ball_x, '@')

    # ball movement and get rewards
    total_rewards += state.move_ball_get_rewards()

curses.endwin()


