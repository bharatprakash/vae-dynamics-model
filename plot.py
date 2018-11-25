from IPython.display import clear_output
import matplotlib.pyplot as plt
import pickle



#nb_pg = pickle.load(open('model_free/results/nb1.p', "rb" ))
b_pg = pickle.load(open('model_free/results/pg_cc_wb2.p', "rb" ))

nb_env = pickle.load(open('results/env_cc_nb.p', "rb" ))

b_env = pickle.load(open('results/env_cc_wb2.p', "rb" ))


#plt.figure(figsize=(20,5))
plt.title('With Bocker')
plt.xlabel('Episodes')
plt.ylabel('Cumulative Catastrophes')
#plt.plot(b_pg, label="Policy Gradient", color="dodgerblue", marker='x', markevery=20)
#plt.plot(b_env, label="Env With Blocker", color="lightblue" )
plt.plot(nb_env, label="Model Based" , color="red", marker='o', markevery=20)
plt.legend()
plt.show()
