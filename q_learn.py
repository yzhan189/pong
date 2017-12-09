from State import *
import random
import time

# discount factor
gamma = 0.7
# learning rate, start at 1, decay as O(1/t), t starts as 1
def alpha(t):
    return 60/(59+t)


# Q(s,a) (up,stay,down)
Q = np.zeros((10369,3))
N = np.zeros((10369,3))
# Q = np.genfromtxt ('Q.csv', delimiter=",")
# N = np.genfromtxt ('N.csv', delimiter=",")


n = 0
t = 0
diff = 0

state = State()
s = state.discretize_get_index()


start_time = time.time()
while n<100:
    # terminal state
    if s == 10369-1:
        n +=1
        Q[s] += [-1,-1,-1]
        # start a new trial
        state = State()
        s = state.discretize_get_index()


    else:
        # get action
        a_t = random.randint(0,2)
        N[s,a_t] += 1

        # take action change state
        if a_t == 0:
            state.move_paddle_up()
        elif a_t == 2:
            state.move_paddle_down()

        # reward of taking a_t at s
        r_t = state.move_ball_get_rewards()

        # get next state
        s_prime =  state.discretize_get_index()

        alph = alpha(N[s,a_t])
        Q[s,a_t] = (1-alph) * Q[s,a_t] + \
                   alph*(r_t + gamma*np.max(Q[s_prime]))
        if s != s_prime:
            diff += 1
        s = s_prime
        t += 1

end_time = time.time()
print(n,t, diff)
print(end_time-start_time)
np.savetxt("Q2000.csv", Q, delimiter=",")
np.savetxt("N2000.csv", N, delimiter=",")



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

print(total_bounce/n)