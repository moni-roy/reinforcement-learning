#!/usr/bin/python3
import argparse
import logging
import sys
import os

import numpy as np

import gym
from gym import wrappers, logger


# state =  [pos(x), vel(xdot)]
MIN_VALS = [-1.2,     -0.07]  # This needs to be changed
MAX_VALS = [0.6,     0.07]  # This needs to be changed
NUM_BINS = [99,     99]  # This needs to be changed
bins = np.array([np.linspace(MIN_VALS[i], MAX_VALS[i], NUM_BINS[i])
                 for i in range(len(MAX_VALS))])

################################################################################
# CS482: this is the function that changes the continuous values of the state of
# the cartpole into a single integer state value, you'll have to adjust this for
# the mountain car task
################################################################################


def discretize_state(states):
    global bins
    discretized_states = 0
    for i, state in enumerate(states):
        discretized_states += ((NUM_BINS[i]+1)**i)*np.digitize(state, bins[i])
    return discretized_states


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)

    parser.add_argument('--env_id', dest='env_id', nargs='?',
                        default='MountainCar-v0', help='Select the environment to run')
    parser.add_argument('--train', dest='train', action='store_true',
                        help=' boolean flag to start training (if this flag is set)')
    parser.add_argument('--test', dest='test', action='store_true',
                        help='boolean flag to start testing (if this flag is set)')
    parser.add_argument('--model', dest='model', type=str, default=None,
                        help='path to learned model')
    args = parser.parse_args()
    train = args.train
    test = args.test

    if not train and not test:
        print("[Warning] Specify train or test flag")
        print("for eg: python3 cart.py --train")
        print("or python3 cart.py --test --model cartpole.npy")

    if test:
        assert args.model is not None,\
            "Error: path to learned model path is not provided."
        if not os.path.exists(args.model):
            print("[Error] invalid model path\nNo such file as '" +
                  args.model+"' found")
            sys.exit(1)

    logger = logging.getLogger()
    formatter = logging.Formatter('[%(asctime)s] %(message)s')
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # You can set the level to logging.DEBUG or logging.WARN if you
    # want to change the amount of output.
    logger.setLevel(logging.INFO)

    env = gym.make(args.env_id)
    outdir = '/tmp/' + 'qagent' + '-results'
    # Comment out the following line to disable rendering of visual which
    # speeds up the training
    env = wrappers.Monitor(env, outdir, write_upon_reset=True, force=True)

    env.seed(0)

    ############################################################################
    # CS482: This initial Q-table size should change to fit the number of
    # actions (columns) and the number of observations (rows)
    ############################################################################

    if train:
        # initialize Q table with zeros
        # The number 9999 needs to be changed every time you chan change NUM_BINS
        Q = np.zeros([9999, env.action_space.n])
    if test:
        # load the saved model(learned Q table)
        Q = np.load(args.model)

    ############################################################################
    # CS482: Here are some of the RL parameters. You have to tune the
    # learning rate (alpha) and the discount factor (gamma)
    ############################################################################

    alpha = 0.5  # This needs to be changed ( same as problem 1)
    gamma = 0.9  # This needs to be changed ( same as problem 1)
    # epsion-greedy params
    eps_start = 0.9
    eps_end = 0.05
    eps_decay = 100000
    if train:
        n_episodes = 50001     # This might need to be changed
        time = 1
        for episode in range(n_episodes):
            tick = 0
            reward = 0
            done = False
            state = env.reset()
            s = discretize_state(state)
            while done != True:
                time += 1
                tick += 1
                action = 0
                ri = -999
                # epsilon-greedy strategy that chooses actions based on
                # exploration or exploitation phase.
                p = np.random.rand()
                epsilon = eps_end+(eps_start-eps_end) * \
                    np.exp(-1.*time/eps_decay)
                if p < epsilon:
                    action = np.random.randint(env.action_space.n)
                    ri = Q[s][action]
                else:
                    # select action that yields max q value
                    for q in range(env.action_space.n):
                        if Q[s][q] > ri:
                            action = q
                            ri = Q[s][q]

                state, reward, done, info = env.step(action)
                sprime = discretize_state(state)
                predicted_value = np.max(Q[sprime])

                ################################################################
                # CS482: Implement the update rule for Q learning here
                ################################################################

                Q[s, action] += alpha*(reward + gamma * predicted_value - Q[s, action])  # This needs to be changed similar to problem 1

                s = sprime

            if episode % 1000 == 0:
                alpha *= .996

            if state[0] < 0.5:
                print("fail ")
            else:
                print("success")
        np.save('car.npy', Q)

    if test:
        import time
        ########################################################################
        # CS482: this part of code relates to testing the performance of
        # the loaded (possibly learned) model
        ########################################################################
        tick = 0
        reward = 0
        done = False
        state = env.reset()
        s = discretize_state(state)
        while done != True:
            tick += 1
            action = 0
            ri = -999
            # select action that yields max q value
            for q in range(env.action_space.n):
                if Q[s][q] > ri:
                    action = q
                    ri = Q[s][q]
            state, reward, done, info = env.step(action)
            sprime = discretize_state(state)
            # render the graphics
            env.render()
            time.sleep(0.01)

            s = sprime
