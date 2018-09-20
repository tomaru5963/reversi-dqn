import collections
import pathlib
import pickle

import numpy as np
import tensorflow as tf

from .player_base import PlayerBase


class DQNPlayer(PlayerBase):

    MEMORY_SIZE = 500
    BATCH_SIZE = 32
    REPLACE_TARGET_STEPS = 300

    def __init__(self, board, name='DQNPlayer', train=False):
        super(DQNPlayer, self).__init__(board, name, train)

        # self.greedy_rate = 0.3
        # self.greedy_rate = 0.05
        self.greedy_rate = 0.1
        # self.discount_rate = 0.9
        self.discount_rate = 1.0

        self.num_states = self.board.NUM_ROWS * self.board.NUM_COLS
        self.num_actions = self.board.NUM_ROWS * self.board.NUM_COLS

        self.train_steps = 0
        self.train_epochs = 0
        self.memory = {}
        for key in ('state', 'action', 'reward', 'observation', 'not_done'):
            self.memory[key] = collections.deque(maxlen=self.MEMORY_SIZE)
        self.last_state = None
        self.last_action = None
        self.initialized = False

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.graph_state = tf.placeholder(tf.float32,
                                              shape=(None, self.num_states),
                                              name='state')

            with tf.variable_scope('q_net'):
                self.graph_q_net = self.build_network(
                    self.graph_state,
                    (self.num_states, self.num_actions),
                    ['q_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
                )

            self.graph_target_state = tf.placeholder(tf.float32,
                                                     shape=(None, self.num_states),
                                                     name='target_state')
            self.graph_target_q = tf.placeholder(tf.float32,
                                                 shape=(None, self.num_actions),
                                                 name='target_q')

            # target (fixed) network
            with tf.variable_scope('target_net'):
                self.graph_target_net = self.build_network(
                    self.graph_target_state,
                    (self.num_states, self.num_actions),
                    ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
                )

            loss = tf.reduce_mean(
                tf.squared_difference(self.graph_target_q, self.graph_q_net)
            )
            self.graph_train_op = tf.train.AdamOptimizer().minimize(loss)

            e_params = tf.get_collection('q_net_params')
            t_params = tf.get_collection('target_net_params')
            self.graph_replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

            self.saver = tf.train.Saver()
            self.sess = tf.Session()

    def save_params(self, path):
        dir_ = pathlib.Path(path)
        if dir_.exists():
            if dir_.is_dir():
                for p in dir_.glob('*'):
                    p.unlink()
                dir_.rmdir()
            else:
                print(f'Cannot save params to {str(dir)} because the file already exists')
                return
        dir_.mkdir()

        params = [self.greedy_rate,
                  self.train_steps,
                  self.train_epochs,
                  self.memory]
        with (dir_ / 'model.pkl').open('wb') as f:
            pickle.dump(params, f)

        self.saver.save(self.sess, str(dir_ / 'model'))

    def load_params(self, path):
        dir_ = pathlib.Path(path)
        if not dir_.exists() or not dir_.is_dir():
            print(f'Cannot restore params from {str(dir)} because the directory does not exist')
            return

        with (dir_ / 'model.pkl').open('rb') as f:
            params = pickle.load(f)
            self.greedy_rate = params[0]
            self.train_steps = params[1]
            self.train_epochs = params[2]
            self.memory = params[3]

        self.saver.restore(self.sess, str(dir_ / 'model'))
        self.initilized = True

    def on_game_started(self, who_am_i):
        self.last_state = None
        self.last_action = None

        if not self.initialized:
            with self.graph.as_default():
                self.sess.run(tf.global_variables_initializer())
            self.initilized = True

    def on_game_finished(self, who_am_i):
        self.train_steps += 1
        self.greedy_rate *= 0.9

    def on_next_move_required(self, who_am_i):
        if self.train:
            return self.on_next_move_required_for_train(who_am_i)

        if self.board.state != self.board.ACTIVE:
            return None

        _, action = self.choose_action(who_am_i)
        return self.action_to_pos(action)

    @staticmethod
    def build_network(state, shape, c_names):
        # weight_init = tf.constant_initializer(1.0)
        weight_init = tf.random_uniform_initializer(-0.5, 0.5)
        baias_init = tf.constant_initializer()

        with tf.variable_scope('layer1'):
            w1 = tf.get_variable('w', shape=(shape[0], int(shape[0] * .7)),
                                 initializer=weight_init,
                                 collections=c_names)
            b1 = tf.get_variable('b', shape=(int(shape[0] * .7),),
                                 initializer=baias_init,
                                 collections=c_names)
            # return tf.matmul(state, w1) + b1
            l1 = tf.nn.sigmoid(tf.matmul(state, w1) + b1)

        with tf.variable_scope('layer2'):
            w2 = tf.get_variable('w', shape=(int(shape[0] * .7), shape[1]),
                                 initializer=weight_init,
                                 collections=c_names)
            b2 = tf.get_variable('b', shape=(shape[1],),
                                 initializer=baias_init,
                                 collections=c_names)
        return tf.nn.sigmoid(tf.matmul(l1, w2) + b2)

    def action_to_pos(self, action):
        return (action // self.board.NUM_COLS, action % self.board.NUM_COLS)

    def pos_to_action(self, pos):
        return pos[0] * self.board.NUM_COLS + pos[1]

    def on_next_move_required_for_train(self, who_am_i):
        board = self.board

        if self.last_action is None:
            state, action = self.choose_action(who_am_i)
            self.last_state = state
            self.last_action = action
            return self.action_to_pos(action)

        if self.train_steps % self.REPLACE_TARGET_STEPS == 0:
            self.sess.run(self.graph_replace_target_op)
        self.train_steps += 1

        reward = 0
        done = False
        if ((board.state == board.WON_X and who_am_i == board.PLAYER_X) or
                (board.state == board.WON_O and who_am_i == board.PLAYER_O)):
            reward = 1
            done = True
        elif ((board.state == board.WON_X and who_am_i == board.PLAYER_O) or
                (board.state == board.WON_O and who_am_i == board.PLAYER_X)):
            reward = 0
            done = True
        elif board.state == board.DRAW:
            reward = 0.5
            done = True

        if done:
            state = np.zeros((self.num_states,))   # dummy
        else:
            state, action = self.choose_action(who_am_i)

        # replay experiences
        self.memory['state'].append(self.last_state)
        self.memory['action'].append(self.last_action)
        self.memory['reward'].append(reward)
        self.memory['observation'].append(state)
        self.memory['not_done'].append(0 if done else 1)

        num_memories = len(self.memory['state'])
        num_batches = min(self.BATCH_SIZE, num_memories)
        batch_indices = np.random.choice(num_memories, num_batches, False)
        states = np.array(self.memory['state'])[batch_indices]
        actions = np.array(self.memory['action'])[batch_indices]
        rewards = np.array(self.memory['reward'])[batch_indices]
        observations = np.array(self.memory['observation'])[batch_indices]
        not_dones = np.array(self.memory['not_done'])[batch_indices]

        # cal q values for evalution net adn target net
        q_values, target_q = self.sess.run(
            [self.graph_q_net, self.graph_target_net],
            feed_dict={self.graph_state: states,
                       self.graph_target_state: observations}
        )

        target = q_values.copy()
        target[np.arange(target.shape[0]), actions] = (
            rewards +
            self.discount_rate * np.max(target_q, axis=1) * not_dones
        )

        # train evaluation net
        self.sess.run(
            self.graph_train_op,
            feed_dict={self.graph_target_q: target,
                       self.graph_state: states}
        )

        if done:
            self.last_state, self.last_action = None, None
            return None
        else:
            self.last_state = state
            self.last_action = action
            return self.action_to_pos(action)

    def choose_action(self, who_am_i):
        placeables = []
        for pos in self.board.available_places[who_am_i]:
            placeables.append(self.pos_to_action(pos))
        assert len(placeables) != 0

        state = self.board_to_state(who_am_i)

        if not self.train or np.random.random() >= self.greedy_rate:
            # predict
            q_values = self.sess.run(self.graph_q_net,
                                     feed_dict={self.graph_state: [state]})
            action_idx = np.argmax(q_values[0, placeables])
            action = placeables[action_idx]
        else:
            action = np.random.choice(placeables)

        return state, action

    def board_to_state(self, player):
        # transform board to state
        # my discs -> 1
        # opponent discs -> -1
        # empty -> 0
        # NB: This algo assumes that board.EMPTY is 0 and negative values
        # never used as player id
        opponent = self.board.PLAYER_O
        if player == self.board.PLAYER_O:
            opponent = self.board.PLAYER_X
        state = np.where(self.board.board == opponent, -1, self.board.board)
        state[state == player] = 1
        return state.reshape((self.num_states,))
