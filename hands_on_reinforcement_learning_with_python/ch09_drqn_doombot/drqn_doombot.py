#########################
#
# DRQN on Doom (via vizdoom) with Deathmatch scenario
# 
#########################



import vizdoom

import tensorflow as tf

import numpy as np
import datetime as dt
import math
import os
import sys



#
### helper funcs
#

def get_final_shape(D, F, S):
    c1 = math.ceil(((D - F + 1) / S))
    p1 = math.ceil((c1 / S))
    c2 = math.ceil(((p1 - F + 1) / S))
    p2 = math.ceil((c2 / S))
    c3 = math.ceil(((p2 - F + 1) / S))
    p3 = math.ceil((c3  / S))
    return int(p3)



#
### DRQN class
#

class DRQN():
    def __init__(self, input_shape, num_actions, initial_learning_rate):
        self.tfcast_type = tf.float32
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.learning_rate = initial_learning_rate

        # cnn hyperparams
        self.filter_size = 5
        self.num_filters = [16, 32, 64]
        self.stride = 2
        self.poolsize = 2
        self.convolution_shape = get_final_shape(input_shape[0], self.filter_size, self.stride) * get_final_shape(input_shape[1], self.filter_size, self.stride) * self.num_filters[2]

        # rnn hyperparams
        self.cell_size = 100
        self.hidden_layer = 50
        self.dropout_prob = [0.3, 0.2]
        
        # optimization hyperparams
        self.loss_decay_rate = 0.96
        self.loss_decay_steps = 180



        #
        ### tf inits
        #

        # cnn inits
        self.input = tf.placeholder(shape=(self.input_shape[0], self.input_shape[1], self.input_shape[2]),
                                    dtype=self.tfcast_type)
        self.target_vector = tf.placeholder(shape=(self.num_actions,  1),
                                    dtype=self.tfcast_type)

        self.feature_map1 = tf.Variable(initial_value=np.random.rand(self.filter_size, self.filter_size, input_shape[2], self.num_filters[0]),
                                    dtype=self.tfcast_type)
        self.feature_map2 = tf.Variable(initial_value=np.random.rand(self.filter_size, self.filter_size, self.num_filters[0], self.num_filters[1]),
                                    dtype=self.tfcast_type)
        self.feature_map3 = tf.Variable(initial_value=np.random.rand(self.filter_size, self.filter_size, self.num_filters[1], self.num_filters[2]),
                                    dtype=self.tfcast_type)

        # rnn inits
        self.h = tf.Variable(initial_value=np.zeros((1, self.cell_size)),
                                    dtype=self.tfcast_type)
        # input -> hidden weights
        self.rU = tf.Variable(initial_value=np.random.uniform(low=-np.sqrt(6.0 / (2 * self.cell_size)),
                                        high=np.sqrt(6.0 / (2 * self.cell_size)),
                                        size=(self.cell_size, self.cell_size)),
                                    dtype=self.tfcast_type)
        # hidden -> hidden weights
        self.rW = tf.Variable(initial_value=np.random.uniform(low=-np.sqrt(6.0 / (self.convolution_shape + self.cell_size)),
                                        high=np.sqrt(6.0 / (self.convolution_shape + self.cell_size)),
                                        size=(self.convolution_shape, self.cell_size)),
                                    dtype=self.tfcast_type)
        # hidden -> output weights
        self.rV = tf.Variable(initial_value=np.random.uniform(low=-np.sqrt(6.0 / (2 * self.cell_size)),
                                        high=np.sqrt(6.0 / (2 * self.cell_size)),
                                        size=(self.cell_size, self.cell_size)),
                                    dtype=self.tfcast_type)
        # bias
        self.rb = tf.Variable(initial_value=np.zeros(self.cell_size),
                                    dtype=self.tfcast_type)
        self.rc = tf.Variable(initial_value=np.zeros(self.cell_size),
                                    dtype=self.tfcast_type)

        # feedforward inits
        self.fW = tf.Variable(initial_value=np.random.uniform(low=-np.sqrt(6.0 / (self.cell_size + self.num_actions)),
                                        high=np.sqrt(6.0 / (self.cell_size + self.num_actions)),
                                        size=(self.cell_size, self.num_actions)),
                                    dtype=self.tfcast_type)        
        self.fb = tf.Variable(initial_value=np.zeros(self.num_actions),
                                    dtype=self.tfcast_type)

        # learn rate
        self.step_count = tf.Variable(initial_value=0, dtype=self.tfcast_type)
        self.learning_rate = tf.train.exponential_decay(self.learning_rate,
                                    self.step_count,
                                    self.loss_decay_steps,
                                    self.loss_decay_steps,
                                    staircase=False)


        #
        ### tf network logic
        # 

        # cnn 1
        self.conv1 = tf.nn.conv2d(input=tf.reshape(self.input, 
                                        shape=(1, self.input_shape[0], self.input_shape[1], self.input_shape[2])),
                                    filter=self.feature_map1,
                                    strides=[1, self.stride, self.stride, 1],
                                    padding='VALID')
        self.relu1 = tf.nn.relu(self.conv1)
        self.pool1 = tf.nn.max_pool(self.relu1,
                                    ksize=[1,self.poolsize, self.poolsize, 1],
                                    strides=[1,self.stride, self.stride, 1],
                                    padding='SAME')
        # cnn 2
        self.conv2 = tf.nn.conv2d(input=self.pool1,
                                    filter=self.feature_map2,
                                    strides=[1, self.stride, self.stride, 1],
                                    padding='VALID')
        self.relu2 = tf.nn.relu(self.conv2)
        self.pool2 = tf.nn.max_pool(self.relu2,
                                    ksize=[1,self.poolsize, self.poolsize, 1],
                                    strides=[1,self.stride, self.stride, 1],
                                    padding='SAME')
        # cnn 3
        self.conv3 = tf.nn.conv2d(input=self.pool2,
                                    filter=self.feature_map3,
                                    strides=[1, self.stride, self.stride, 1],
                                    padding='VALID')
        self.relu3 = tf.nn.relu(self.conv3)
        self.pool3 = tf.nn.max_pool(self.relu3,
                                    ksize=[1,self.poolsize, self.poolsize, 1],
                                    strides=[1,self.stride, self.stride, 1],
                                    padding='SAME')

        # dropout + reshaping
        self.drop1 = tf.nn.dropout(self.pool3,
                                    self.dropout_prob[0])
        self.reshaped_input = tf.reshape(self.drop1, 
                                    shape=[1,-1])

        # rnn
        self.h = tf.tanh(tf.matmul(self.reshaped_input, self.rW) + tf.matmul(self.h, self.rU) + self.rb)
        self.o = tf.nn.softmax(tf.matmul(self.h, self.rV) + self.rc)
        # rnn dropout
        self.drop2 = tf.nn.dropout(self.o,
                                    self.dropout_prob[1])

        # feedforward layer
        self.output = tf.reshape(tf.matmul(self.drop2, self.fW) + self.fb,
                                    shape=[-1,1])
        self.prediction = tf.argmax(self.output)


        # loss + optimizer
        self.loss = tf.reduce_mean(tf.square(self.target_vector - self.output))
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.gradients = self.optimizer.compute_gradients(self.loss)
        self.update = self.optimizer.apply_gradients(self.gradients)
        self.parameters = (self.feature_map1, self.feature_map2, self.feature_map3,
                        self.rW, self.rU, self.rV, self.rb, self.rc,
                        self.fW, self.fb)



#
### experience buffer
#

class ExperienceReplay():
    def __init__(self, buffer_size):
        self.buffer = []
        self.buffer_size = buffer_size

        self.is_buffer_at_cap = False

    
    def append_to_buffer(self, memory_tuple):
        if not self.is_buffer_at_cap:
            if len(self.buffer) > self.buffer_size:
                self.is_buffer_at_cap = True
        
        if self.is_buffer_at_cap:
            self.buffer.pop(0)
        
        self.buffer.append(memory_tuple)
    

    def sample(self, n):
        memories = []
        for _i in range(n):
            memory_index = np.random.randint(0, len(self.buffer))
            memories.append(self.buffer[memory_index])
        
        return memories



#
### training
#

def train(episodes, episode_len, learning_rate, scenario='deathmatch.cfg', scenario_dir='', map_path='map02', render=False):
    # hardcoded params
    discount = .99
    update_freq = 5
    store_freq = 50
    print_freq = 1000
    save_freq = 100

    # inits
    total_reward = 0
    total_loss = 0
    q_value_old = 0
    rewards = []
    losses = []

    # game settings
    dg = vizdoom.DoomGame()
    dg.set_doom_scenario_path(scenario_dir + scenario)
    dg.set_doom_map(map_path)

    dg.set_screen_resolution(vizdoom.ScreenResolution.RES_256X160)
    dg.set_screen_format(vizdoom.ScreenFormat.RGB24)
    input_shape = (160,256,3)

    dg.set_render_hud(False)
    dg.set_render_minimal_hud(False)
    dg.set_render_crosshair(False)
    dg.set_render_weapon(True)
    dg.set_render_decals(False)
    dg.set_render_particles(False)
    dg.set_render_effects_sprites(False)
    dg.set_render_messages(False)
    dg.set_render_corpses(False)
    dg.set_render_screen_flashes(True)

    dg.add_available_game_variable(vizdoom.GameVariable.AMMO0)
    dg.add_available_game_variable(vizdoom.GameVariable.HEALTH)
    dg.add_available_game_variable(vizdoom.GameVariable.KILLCOUNT)

    dg.set_episode_timeout(6 * episode_len)
    dg.set_episode_start_time(10)
    dg.set_window_visible(render)
    dg.set_sound_enabled(False)

    dg.set_mode(vizdoom.Mode.PLAYER)

    dg.set_living_reward(0)

    dg.add_available_button(vizdoom.Button.MOVE_LEFT)
    dg.add_available_button(vizdoom.Button.MOVE_RIGHT)
    dg.add_available_button(vizdoom.Button.TURN_LEFT)
    dg.add_available_button(vizdoom.Button.TURN_RIGHT)
    dg.add_available_button(vizdoom.Button.MOVE_FORWARD)
    dg.add_available_button(vizdoom.Button.MOVE_BACKWARD)
    dg.add_available_button(vizdoom.Button.ATTACK)
    dg.add_available_button(vizdoom.Button.TURN_LEFT_RIGHT_DELTA, 90)
    dg.add_available_button(vizdoom.Button.LOOK_UP_DOWN_DELTA, 90)

    # init action space
    actions = np.zeros((dg.get_available_buttons_size(), dg.get_available_buttons_size()))
    count = 0
    for i in actions:
        i[count] = 1
        count += 1
    actions = actions.astype(int).tolist()

    # game init
    dg.init()


    # init drqns & buffer
    drqn_action = DRQN(input_shape, dg.get_available_buttons_size() - 2, learning_rate)
    drqn_target = DRQN(input_shape, dg.get_available_buttons_size() - 2, learning_rate)
    experiences = ExperienceReplay(1000)

    # model saver
    saver = tf.train.Saver({v.name: v for v in drqn_action.parameters}, max_to_keep=1)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        t0 = dt.datetime.now()

        for episode in range(episodes):
            # start episode
            t1 = dt.datetime.now()
            dg.new_episode()
            max_frame = episode_len
            for frame in range(episode_len):
                # get state (pixels)
                state = dg.get_state()
                s = state.screen_buffer

                # select action
                a = drqn_action.prediction.eval(feed_dict = {drqn_action.input: s})[0]
                action = actions[a]

                # take action & store reward
                reward = dg.make_action(action)
                total_reward += reward

                # break if done
                if dg.is_episode_finished():
                    max_frame = frame
                    break
                
                # store transition in buffer
                if (frame % store_freq) == 0:
                    experiences.append_to_buffer((s, action, reward))

                # sample & update networks
                if (frame % update_freq) == 0:
                    memory = experiences.sample(1)
                    mem_frame = memory[0][0]
                    mem_reward = memory[0][2]

                    # train
                    Q1 = drqn_action.output.eval(feed_dict = {drqn_action.input: mem_frame})
                    Q2 = drqn_target.output.eval(feed_dict = {drqn_target.input: mem_frame})

                    # set learning rate
                    learning_rate = drqn_action.learning_rate.eval()

                    # calc q value
                    q_target = q_value_old + learning_rate * (mem_reward + (discount * Q2) - q_value_old )
                    q_value_old = q_target

                    # compute loss
                    loss = drqn_action.loss.eval(feed_dict = {drqn_action.target_vector: q_target, drqn_action.input: mem_frame})
                    total_loss += loss

                    # update
                    drqn_action.update.run(feed_dict = {drqn_action.target_vector: q_target, drqn_action.input: mem_frame})
                    drqn_target.update.run(feed_dict = {drqn_target.target_vector: q_target, drqn_target.input: mem_frame})

            
            rewards.append((episode, total_reward))
            losses.append((episode, total_loss))

            print('EPISODE {}\treward={}\tlength={}\tloss={:.5f}\tlearn rate={:.5f}\ttime={}'.format(episode, 
                total_reward,
                max_frame, 
                total_loss,
                learning_rate,
                dt.datetime.now() - t1))
            
            total_reward = 0
            total_loss = 0

            if (episode % save_freq) == 0:
                sp = saver.save(sess, './model/drqn_doombot.ckpt')
                print('DRQN SAVED TO:{}\ttotal train time: {}'.format(sp,
                    dt.datetime.now() - t0))

        sp = saver.save(sess, './model/drqn_doombot_final.ckpt')
        print('TRAINING COMPLETE!\tepisodes: {}\ttotal train time: {}'.format(episodes,
            t0 - dt.datetime.now()))



if __name__ == "__main__":
    train(episodes=10000,
        episode_len=300,
        learning_rate=0.01,
        scenario_dir='/Users/jj/anaconda3/lib/python3.6/site-packages/vizdoom/scenarios/',
        render=False)