from State import *

state = State()

class Left_Paddle:
    y = 0.4
    paddle_height = 0.2
    speed = 0.02

    def move(self,ball_y):
        if ball_y > self.y+0.1:
            self.y += self.speed
            self.y = min(self.y,0.8)
        if ball_y < self.y+0.1:
            self.y -= self.speed
            self.y = max(self.y, 0)