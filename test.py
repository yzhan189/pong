from State import *
Q = np.genfromtxt ('Q2000.csv', delimiter=",")

n = 0
total_bounce = 0
state = State()
while n<1000:
    index = state.discretize_get_index()
    action = np.argmax(Q[index])
    if action ==0:
        state.move_paddle_up()
    elif action ==2:
        state.move_paddle_down()
    reward = state.move_ball_get_rewards()
    if reward==-1:
        state = State()
        n += 1
    else:
        total_bounce += reward

print("Total test games: " + str(n))
print("Average bounces: "+ str(total_bounce/n))

print(total_bounce/n)