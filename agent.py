import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.layers.merge import Add, Multiply
from keras.optimizers import Adam
import keras.backend as K

import tensorflow as tf

import random
from collections import deque

INT_TO_AMOUNT = {
    0:  5,
    1: 10,
    2: 25,
    3: 50,
}

def get_max_bet(user_money):
    sorted_values = sorted(INT_TO_AMOUNT, reverse=True)
    for i in sorted_values:
        if user_money >= INT_TO_AMOUNT[i]:
            return i

class ActorCritic:
    def __init__(self, env, sess):
        self.env = env
        self.sess = sess

        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = .995
        self.gamma = .95
        self.tau = .125

        self.memory = deque(maxlen=2000)

        # ===================================================================== #
        #                              Better Model                             #
        # ===================================================================== #

        self.better_state_input, self.better_model = self.create_better_model()
        _, self.target_better_model = self.create_better_model()

        self.better_critic_grad = tf.placeholder(tf.float32, [None, 12])  # where we will feed de/dC (from critic)

        better_weights = self.better_model.trainable_weights
        self.better_grads = tf.gradients(self.better_model.output, better_weights,
                                         -self.better_critic_grad)  # dC/dA (from actor)

        grads = zip(self.better_grads, better_weights)
        self.better_optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)

        # ===================================================================== #
        #                              Actor Model                              #
        # ===================================================================== #

        self.actor_state_input, self.actor_model = self.create_actor_model()
        _, self.target_actor_model = self.create_actor_model()

        self.actor_critic_grad = tf.placeholder(tf.float32, [None, 15])  # where we will feed de/dC (from critic)

        actor_model_weights = self.actor_model.trainable_weights
        self.actor_grads = tf.gradients(self.actor_model.output, actor_model_weights,
                                        -self.actor_critic_grad)  # dC/dA (from actor)

        grads = zip(self.actor_grads, actor_model_weights)
        self.actor_optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)

        # ===================================================================== #
        #                              Critic Model                             #
        # ===================================================================== #

        self.pre_state_input, self.pre_action_input, self.state_input, self.action_input, self.recap_input, self.critic_model = self.create_critic_model()
        _, _, _, _, _, self.target_critic_model = self.create_critic_model()

        self.critic_grads = tf.gradients(self.critic_model.output, self.pre_action_input, self.action_input)  # where we calcaulte de/dC for feeding above

        # Initialize for later gradient calculations
        self.sess.run(tf.initialize_all_variables())

    # ========================================================================= #
    #                              Model Definitions                            #
    # ========================================================================= #

    def create_better_model(self):
        state_input = Input(shape=(12,))
        h1 = Dense(24, activation='relu')(state_input)
        h2 = Dense(48, activation='relu')(h1)
        h3 = Dense(24, activation='relu')(h2)
        output = Dense(1, activation='relu')(h3)

        model = Model(input=state_input, output=output)
        adam = Adam(lr=0.001)
        model.compile(loss="mse", optimizer=adam)
        return state_input, model

    def create_actor_model(self):
        state_input = Input(shape=(15,))
        h1 = Dense(24, activation='relu')(state_input)
        h2 = Dense(48, activation='relu')(h1)
        h3 = Dense(24, activation='relu')(h2)
        output = Dense(3, activation='relu')(h3)

        model = Model(input=state_input, output=output)
        adam = Adam(lr=0.001)
        model.compile(loss="mse", optimizer=adam)
        return state_input, model

    def create_critic_model(self):
        # initial state + money1 + players
        pre_state_input = Input(shape=(12,))
        pre_state_h1 = Dense(24, activation='relu')(pre_state_input)
        pre_state_h2 = Dense(48)(pre_state_h1)

        pre_action_input = Input(shape=(1,))
        pre_action_h1 = Dense(48)(pre_action_input)

        pre_merged = Add()([pre_state_h2, pre_action_h1])
        pre_merged_h1 = Dense(24, activation='relu')(pre_merged)

        # new state + money2 + players + bet + user_total + dealer_total
        state_input = Input(shape=(15,))
        state_h1 = Dense(24, activation='relu')(state_input)
        state_h2 = Dense(48)(state_h1)

        action_input = Input(shape=(3,))
        action_h1 = Dense(48)(action_input)

        merged = Add()([state_h2, action_h1])
        merged_h1 = Dense(24, activation='relu')(merged)

        # final state + final user_total + final dealer_total + reward
        recap_input = Input(shape=(13,))
        recap_h1 = Dense(24, activation='relu')(recap_input)
        recap_h2 = Dense(48)(recap_h1)

        merged_input = Add()([pre_merged_h1, merged_h1, recap_h2])
        merged_final1 = Dense(24, activation='relu')(merged_input)
        merged_total2 = Dense(12, activation='relu')(merged_final1)

        output = Dense(2, activation='relu')(merged_total2)
        model = Model(input=[pre_state_input, pre_action_input, state_input, action_input, recap_input], output=output)

        adam = Adam(lr=0.001)
        model.compile(loss="mse", optimizer=adam)
        return pre_state_input, pre_action_input, state_input, action_input, recap_input, model

    # ========================================================================= #
    #                               Model Training                              #
    # ========================================================================= #

    def remember(self, pre_states, bets, new_states, actions, reward, final_states):
        for pre_state, bet, new_state, action, reward, final_state in zip(pre_states, bets, new_states, actions, final_states):
            self.memory.append([pre_state, bet, new_state, action, reward, final_state])

    def _train_better(self, samples):
        for sample in samples:
            pre_state, bet, new_state, action, reward, final_state = sample
            predicted_action = self.better_model.predict(pre_state)
            grads = self.sess.run(self.critic_grads, feed_dict={
                self.critic_state_input: pre_state,
                self.critic_action_input: predicted_action
            })[0]

            self.sess.run(self.optimize, feed_dict={
                self.better_state_input: pre_state,
                self.better_critic_grad: grads
            })

    def _train_actor(self, samples):
        for sample in samples:
            pre_state, bet, new_state, action, reward, final_state = sample
            predicted_action = self.actor_model.predict(pre_state)
            grads = self.sess.run(self.critic_grads, feed_dict={
                self.critic_state_input: pre_state,
                self.critic_action_input: predicted_action
            })[0]

            self.sess.run(self.optimize, feed_dict={
                self.actor_state_input: pre_state,
                self.actor_critic_grad: grads
            })

    def _train_critic(self, samples):
        for sample in samples:
            pre_state, bet, new_state, action, reward, final_state = sample
            target_action = self.target_actor_model.predict(new_state)
            future_reward = self.target_critic_model.predict(
                [new_state, target_action])[0][0]
            reward += self.gamma * future_reward
            self.critic_model.fit([pre_state, action], reward, verbose=0)

    def train(self):
        batch_size = 32
        if len(self.memory) < batch_size:
            return

        rewards = []
        samples = random.sample(self.memory, batch_size)
        self._train_critic(samples)
        self._train_better(samples)
        self._train_actor(samples)

    # ========================================================================= #
    #                         Target Model Updating                             #
    # ========================================================================= #

    def _update_better_target(self):
        better_model_weights = self.better_model.get_weights()
        better_target_weights = self.target_better_model.get_weights()

        for i in range(len(better_target_weights)):
            better_target_weights[i] = better_model_weights[i]
        self.target_critic_model.set_weights(better_target_weights)

    def _update_actor_target(self):
        actor_model_weights = self.actor_model.get_weights()
        actor_target_weights = self.target_actor_model.get_weights()

        for i in range(len(actor_target_weights)):
            actor_target_weights[i] = actor_model_weights[i]
        self.target_critic_model.set_weights(actor_target_weights)

    def _update_critic_target(self):
        critic_model_weights = self.critic_model.get_weights()
        critic_target_weights = self.critic_target_model.get_weights()

        for i in range(len(critic_target_weights)):
            critic_target_weights[i] = critic_model_weights[i]
        self.critic_target_model.set_weights(critic_target_weights)

    def update_target(self):
        self._update_better_target()
        self._update_actor_target()
        self._update_critic_target()

    # ========================================================================= #
    #                              Model Predictions                            #
    # ========================================================================= #

    def make_bet(self, cur_state, user_money):
        self.epsilon *= self.epsilon_decay
        if np.random.random() < self.epsilon:
            return random.randint(0, user_money)
        return self.better_model.predict(cur_state)

    def make_action(self, cur_state):
        self.epsilon *= self.epsilon_decay
        if np.random.random() < self.epsilon:
            return random.randint(0, 2)
        return self.actor_model.predict(cur_state)


class ActorCriticCombined:
    pass

class DQL:
    def __init__(self):
        self.memory = deque(maxlen=2000)

        self.gamma = 0.85
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.005
        self.tau = .125

        self.better_model = self.create_better()
        self.target_better_model = self.create_better()

        self.actor_model = self.create_actor()
        self.target_actor_model = self.create_actor()

    def create_better(self):
        model = Sequential()
        model.add(Dense(24, input_shape=(12,), activation="relu"))
        model.add(Dense(48, activation="relu"))
        model.add(Dense(48, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(4))
        model.compile(loss="mean_squared_error", optimizer=Adam(lr=self.learning_rate))
        return model

    def create_actor(self):
        model = Sequential()
        model.add(Dense(24, input_shape=(15,), activation="relu"))
        model.add(Dense(48, activation="relu"))
        model.add(Dense(48, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(3))
        model.compile(loss="mean_squared_error", optimizer=Adam(lr=self.learning_rate))
        return model

    def bet(self, state, money):
        state = np.expand_dims(state, axis=0)
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            print("RANDOM")
            return random.randint(0, get_max_bet(money))

        better_model = self.better_model
        bet_args = better_model.predict(state)
        bets = np.argsort(bet_args[0])[::-1]

        for i in bets:
            if money >= INT_TO_AMOUNT[i]:
                return i

    def act(self, state, money, bet, initial):
        state = np.expand_dims(state, axis=0)
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            print("RANDOM")
            if initial and money >= bet:
                return random.randint(0, 2)
            else:
                return random.randint(0, 1)
        actor_model = self.actor_model
        action_args = actor_model.predict(state)
        actions = np.argsort(action_args[0])[::-1]
        for i in actions:
            if i == 2 and (not initial or money < bet):
                continue
            return i

    def remember(self, pre_states, bets, new_states, actions, rewards, final_states, dones):
        for pre_state, bet, new_state, action, reward, final_state, done in zip(pre_states, bets, new_states, actions, rewards, final_states, dones):
            self.memory.append((pre_state, bet, new_state, action, reward, final_state[:12], done))

    def replay(self):
        print("REPLAYING")
        batch_size = 32
        if len(self.memory) < batch_size:
            return

        samples = random.sample(self.memory, batch_size)

        for sample in samples:
            pre_state, bet, new_state, action, reward, final_state, done = sample
            pre_state = np.expand_dims(pre_state, axis=0)
            new_state = np.expand_dims(new_state, axis=0)
            final_state = np.expand_dims(final_state, axis=0)

            target_bet = self.target_better_model.predict(pre_state)
            target_act = self.target_actor_model.predict(new_state)
            if done:
                target_bet[0][bet] = reward
                target_act[0][action] = reward
            else:
                # Target Better
                Q_future_bet = max(self.target_actor_model.predict(new_state)[0])
                target_bet[0][bet] = reward + Q_future_bet * self.gamma

                # Target Actor
                Q_future_act = max(self.target_better_model.predict(final_state)[0])
                target_act[0][action] = reward + Q_future_act * self.gamma
            self.better_model.fit(pre_state, target_bet, epochs=1, verbose=0)
            self.actor_model.fit(new_state, target_act, epochs=1, verbose=0)

    def target_train(self):
        bet_weights = self.better_model.get_weights()
        target_bet_weights = self.target_better_model.get_weights()
        for i in range(len(target_bet_weights)):
            target_bet_weights[i] = bet_weights[i] * self.tau + target_bet_weights[i] * (1 - self.tau)
        self.target_better_model.set_weights(target_bet_weights)

        actor_weights = self.actor_model.get_weights()
        target_actor_weights = self.target_actor_model.get_weights()
        for i in range(len(target_actor_weights)):
            target_actor_weights[i] = actor_weights[i] * self.tau + target_actor_weights[i] * (1 - self.tau)
        self.target_actor_model.set_weights(target_actor_weights)


    def save_model(self, fn):
        self.model.save(fn)