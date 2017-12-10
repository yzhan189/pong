from q_learn import *

result = np.zeros((10,10,10))
i = 0
j = 0
k = 0
for gamma in [0.5]:#np.linspace(0.1,1.1,10):
    for alpha_C in [100]:#np.linspace(20,500,10):
        for N_e in np.linspace(10,100,10):
            #result[i,j,k] =
            q_learn(gamma, alpha_C,N_e)
    #         k += 1
    #     j+=1
    # i+=1
