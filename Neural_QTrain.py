import gym
import tensorflow as tf
import numpy as np
import random

# General Parameters
# -- DO NOT MODIFY --
ENV_NAME = 'CartPole-v0'
EPISODE = 200000  # Episode limitation
STEP = 200  # Step limitation in an episode
TEST = 10  # The number of tests to run every TEST_FREQUENCY episodes
TEST_FREQUENCY = 100  # Num episodes to run before visualizing test accuracy

# TODO: HyperParameters
GAMMA =0.9  # discount factor
INITIAL_EPSILON =0.6  # starting value of epsilon
FINAL_EPSILON =0.1  # final value of epsilon
EPSILON_DECAY_STEPS =100  # decay period
HIDDEN_NODES = 100

REPLAY_SIZE = 1000  # experience replay buffer size

BATCH_SIZE = 512  # size of minibatch

global replay_buffer   # 这个buffer是global的

replay_buffer = []
# Create environment
# -- DO NOT MODIFY --
env = gym.make(ENV_NAME)
epsilon = INITIAL_EPSILON
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.n

# Placeholders
# -- DO NOT MODIFY --
state_in = tf.placeholder("float", [None, STATE_DIM])
action_in = tf.placeholder("float", [None, ACTION_DIM])
target_in = tf.placeholder("float", [None])


# TODO: Define Network Graph


def deep_nn(state_in, state_dim, hidden_nodes, action_dim):
    W1 = tf.get_variable("W1", shape=[state_dim, hidden_nodes],)
    b1 = tf.get_variable("b1", shape=[1, hidden_nodes], initializer = tf.constant_initializer(0.0))
    # Define W and b of the second layer
    # W2 = tf.get_variable("W2", shape=[hidden_nodes, hidden_nodes],)
    # b2 = tf.get_variable("b2", shape=[1, hidden_nodes], initializer = tf.constant_initializer(0.0))
    # # Define W and b of the third layer
    W3 = tf.get_variable("W3", shape=[hidden_nodes, action_dim])
    b3 = tf.get_variable("b3", shape=[1, action_dim], initializer = tf.constant_initializer(0.0))
    # Layer1
    logits_layer1 = tf.matmul(state_in, W1) + b1
    output_layer1 = tf.tanh(logits_layer1)  # tf.sigmoid(logits_layer1) or tf.tanh(logits_layer1) or ... ?
    # # Layer2
    # logits_layer2 = tf.matmul(output_layer1, W2) + b2
    # output_layer2 = tf.tanh(logits_layer2)  # tf.sigmoid(logits_layer2) or tf.tanh(logits_layer2) or ... ?
    # Layer3
    logits_layer3 = tf.matmul(output_layer1, W3) + b3
    output_layer3 = logits_layer3  # tf.sigmoid(logits_layer3) or tf.tanh(logits_layer3) or ... ?
    
    return output_layer3

# TODO: Network outputs
q_values =deep_nn(state_in, STATE_DIM, HIDDEN_NODES, ACTION_DIM)
q_action =tf.reduce_sum(tf.multiply(q_values, action_in), reduction_indices=1)

# TODO: Loss/Optimizer Definition
loss =tf.reduce_sum(tf.square(target_in - q_action))
optimizer =tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
train_loss_summary_op = tf.summary.scalar("TrainingLoss", loss)
# Start session - Tensorflow housekeeping
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())


#更新buffer中的训练数据
def update_replay_buffer(replay_buffer, state, action, reward, next_state, done,action_dim):
    one_hot_action = np.zeros(action_dim)

    replay_buffer.append((state, action, reward, next_state, done))
    if len(replay_buffer) > REPLAY_SIZE:
        replay_buffer.pop(0)
    return None

#喂数据的

def do_train_step(replay_buffer, state_in, action_in, target_in,q_values, q_selected_action, loss, optimise_step,train_loss_summary_op, batch_presentations_count):
    minibatch = random.sample(replay_buffer, BATCH_SIZE) #随机的在buffer中获取 bitchsize长度的数据
    target_batch, state_batch, action_batch = get_train_batch(q_values, state_in, minibatch)
    summary, _ = session.run([train_loss_summary_op, optimise_step], feed_dict={
        target_in: target_batch,
        state_in: state_batch,
        action_in: action_batch
    })

    #writer.add_summary(summary, batch_presentations_count) # tsboard的函数



def get_train_batch(q_values, state_in, minibatch):  #在喂数据的函数中调用的 类似于我们的组函数

    state_batch = [data[0] for data in minibatch]

    action_batch = [data[1] for data in minibatch]

    reward_batch = [data[2] for data in minibatch]

    next_state_batch = [data[3] for data in minibatch]

    target_batch = []

    Q_value_batch = q_values.eval(feed_dict={

        state_in: next_state_batch

    })
    global GAMMA

    for i in range(0, BATCH_SIZE):

        sample_is_done = minibatch[i][4]

        if sample_is_done:

            target_batch.append(reward_batch[i])

        else:

            # TO IMPLEMENT: set the target_val to the correct Q value update

            target_val = reward_batch[i] + GAMMA * np.max(Q_value_batch[i])

            target_batch.append(target_val)

    return target_batch, state_batch, action_batch

# -- DO NOT MODIFY ---
def explore(state, epsilon):  #get action
    """
    Exploration function: given a state and an epsilon value,
    and assuming the network has already been defined, decide which action to
    take using e-greedy exploration based on the current q-value estimates.
    """
    Q_estimates = q_values.eval(feed_dict={
        state_in: [state]
    })
    if random.random() <= epsilon:
        action = random.randint(0, ACTION_DIM - 1)
    else:
        action = np.argmax(Q_estimates)
    one_hot_action = np.zeros(ACTION_DIM)
    one_hot_action[action] = 1
    return one_hot_action

batch_presentations_count=0
# Main learning loop
for episode in range(EPISODE):

    # initialize task
    state = env.reset()

    # Update epsilon once per episode
    epsilon -= epsilon / EPSILON_DECAY_STEPS

    # Move through env according to e-greedy policy
    for step in range(STEP):
        action = explore(state, epsilon)
        next_state, reward, done, _ = env.step(np.argmax(action))
        
        # TODO: Calculate the target q-value.
        # hint1: Bellman
        # hint2: consider if the episode has terminated
        update_replay_buffer(replay_buffer, state, action, reward,next_state, done, ACTION_DIM)
        

        # Do one training step
        if (len(replay_buffer) > BATCH_SIZE):
            do_train_step(replay_buffer, state_in, action_in, target_in,q_values, q_action, loss, optimizer,train_loss_summary_op, batch_presentations_count)
            batch_presentations_count += 1


        # Update
        state = next_state
        if done:
            break




        

    # Test and view sample runs - can disable render to save time
    # -- DO NOT MODIFY --
    if (episode % TEST_FREQUENCY == 0 and episode != 0):
        total_reward = 0
        for i in range(TEST):
            state = env.reset()
            for j in range(STEP):
                # env.render()
                action = np.argmax(q_values.eval(feed_dict={
                    state_in: [state]
                }))
                state, reward, done, _ = env.step(action)
                total_reward += reward
                if done:
                    break
        ave_reward = total_reward / TEST
        print('episode:', episode, 'epsilon:', epsilon, 'Evaluation '
                                                        'Average Reward:', ave_reward)

env.close()
