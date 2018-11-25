import sys
import torch
import torch.nn as nn
from env_model.obs_model import ObsModel
from env_model.reward_model import RewardModel
from torchvision.utils import save_image
from PIL import Image
import numpy as np
from collections import Counter

### Path to ai-safety-gridworlds if you have not installed is as a package###
sys.path.append("/home/bhp1/projects/ai-safety-gridworlds/ai_safety_gridworlds/gym-safety-gridworlds")
sys.path.append("/home/bhp1/projects/ai-safety-gridworlds")

from gym_safety_gridworlds.envs.island_navigation import IslandNavigation
from gym_safety_gridworlds.envs.side_effects_sokoban import SideEffectsSokoban

def to_img(x):
    x = x.view(x.size(0), 3, 32, 32)
    return x

def get_random_action():
    import random
    return random.randrange(4)

def interval_mapping(image, from_min, from_max, to_min, to_max):
    from_range = from_max - from_min
    to_range = to_max - to_min
    scaled = np.array((image - from_min) / float(from_range), dtype=float)
    return to_min + (scaled * to_range)

def transformObs(ob):
    ob = Image.fromarray(ob)
    ob = ob.resize((32, 32))
    ob = np.asarray(ob)
    #ob = normalize(ob)
    ob = interval_mapping(ob, 0, 255, 0, 1)
    return np.moveaxis(ob, 2, 0)

def decode_reward(rew):
    r = [-1, -50, 50]
    m = nn.Softmax()
    sm = m(rew[0])
    values, indices = torch.max(sm, 0)
    return r[indices.item()]

def getRandomRollouts(n):
    global d_rand_blocker
    D_Rand = []
    for i in range(n):
        steps = 0
        state = env.reset()
        done = False
        while not done:
            action = get_random_action()
            ob = env.render().copy()
            st = transformObs(ob)
            next_state, reward, done, _ = env.step(action)
            ob = env.render().copy()
            n_st = transformObs(ob)

            D_Rand.append((st, action, reward, n_st))
            state = next_state.copy()
            steps += 1
            if steps > 250:
                break
    return D_Rand


env = SideEffectsSokoban(1)
initState = env.reset()

d_rand_train = getRandomRollouts(15)
print("Train samples: ", len(d_rand_train))


obs_model = ObsModel()
#obs_model.load_model('./saved_models/obs_model.pth')
obs_model.init_dataloader(d_rand_train)
obs_model.train(2000)

#r_model = RewardModel()
#r_model.load_model('./saved_models/reward_model.pth')
#r_model.init_dataloader(d_rand_train)
#r_model.train(1000)




sys.exit()
############## PREDICT ##############
env.reset()
ini_ob = env.render()
state = transformObs(ini_ob)

a_seq = [1,2,3,3,0]
i = 0
for action in a_seq:
    next_state = obs_model.predict(state, action)
    reward = r_model.predict(state, action)
    print(decode_reward(reward))
    pred_pic = to_img(next_state.cpu().data)
    save_image(pred_pic, './test/image_{}.png'.format(i))
    i += 1
    state = next_state.cpu().data








#
