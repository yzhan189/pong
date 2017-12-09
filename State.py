import numpy as np

# on [0,1]
class State:
    paddle_height = 0.2
    paddle_velocity = 0.04

    def __init__(self):
        self.ball_x = 0.5
        self.ball_y = 0.5
        self.velocity_x = 0.03
        self.velocity_y = 0.01
        # this one is for actual position
        self.paddle_y = 0.5 - self.paddle_height/2

    def move_paddle_down(self):
        self.paddle_y = min(1 - self.paddle_height, self.paddle_y+self.paddle_velocity)

    def move_paddle_up(self):
        self.paddle_y = max(0, self.paddle_y-self.paddle_velocity)



    def move_ball_get_rewards(self,left_paddle = None):
        reward = 0

        # move
        self.ball_y += self.velocity_y
        self.ball_x += self.velocity_x

        # set velocity and bounce
        # top wall
        if self.ball_y <= 0:
            self.ball_y = -self.ball_y
            self.velocity_y = -self.velocity_y
        # bottom wall:
        elif self.ball_y >= 1:
            self.ball_y = 2 - self.ball_y
            self.velocity_y = -self.velocity_y
        # left wall
        elif self.ball_x <= 0:
            if left_paddle == None or (self.ball_y >= left_paddle.y and self.ball_y <= left_paddle.y+left_paddle.paddle_height):
                self.ball_x = -self.ball_x
                self.velocity_x = -self.velocity_x


        # right side
        elif self.ball_x >= 1:
            # paddle
            if self.ball_y >= self.paddle_y and self.ball_y <= self.paddle_y+self.paddle_height:
                self.ball_x = 2 - self.ball_x
                self.velocity_x = -self.velocity_x + (-0.015+np.random.random_sample()*0.03)
                self.velocity_y = self.velocity_y + (-0.03+np.random.random_sample()*0.06)
                reward = 1
            # out of board
            else:
                reward = -1
                # re-start
                self.ball_x = 0.5
                self.ball_y = 0.5
                self.velocity_x = 0.03
                self.velocity_y = 0.01


        # make sure the abs value is greater than 0.03
        if self.velocity_x <= 0:
            self.velocity_x = min(-0.03, self.velocity_x)
        else:
            self.velocity_x = max(0.03, self.velocity_x)

        # make sure v is less than 1
        self.velocity_y = min(0.999,self.velocity_y)
        self.velocity_x = min(0.999,self.velocity_x)

        return reward



    def discretize_get_index(self):
        # discrete is for q learning
        discrete_paddle_y = np.floor(12 * self.paddle_y / (1 - self.paddle_height))
        discrete_paddle_y = min(discrete_paddle_y,11)
        discrete_ball_x = int(self.ball_x*12)
        discrete_ball_y = int(self.ball_y*12)
        discrete_velocity_x = 1 if self.velocity_x > 0 else -1
        if abs(self.velocity_y) < 0.015:
            discrete_velocity_y = 0
        elif self.velocity_y > 0:
            discrete_velocity_y = 1
        else:
            discrete_velocity_y = -1

        # set state index
        # get out off board
        if self.ball_x >1 and (self.ball_y < self.paddle_y or self.ball_y > self.paddle_y + self.paddle_height):
            index = 10369 - 1
        else:
            index = int(discrete_ball_x + 12*discrete_ball_y +\
                         144*((discrete_velocity_x+1)//2) +\
                         144*2*(discrete_velocity_y+1) +\
                         144*2*3*discrete_paddle_y)

        return index

