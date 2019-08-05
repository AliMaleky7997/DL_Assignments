import tensorflow as tf
import gym
import numpy as np


# You must add few lines of code and change all -1s

class Agent:
    def __init__(self, learning_rate):
        # Build the network to predict the correct action
        tf.reset_default_graph()
        input_dimension = 4
        hidden_dimension = 20
        self.input = tf.placeholder(dtype=tf.float32, shape=[1, input_dimension], name='X')
        hidden_layer = tf.layers.dense(self.input, hidden_dimension,
                                       kernel_initializer=tf.initializers.random_normal())
        logits = tf.layers.dense(hidden_layer, 2, kernel_initializer=tf.initializers.random_normal())

        # Sample an action according to network's output
        # use tf.multinomial and sample one action from network's output
        self.action = tf.multinomial(logits, 1)

        # Optimization according to policy gradient algorithm
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(self.action, 2), logits=logits)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)  # use one of tensorflow optimizers
        grads_vars = self.optimizer.compute_gradients(
            cross_entropy)  # gradient of current action w.r.t. network's variables
        self.gradients = [grad for grad, var in grads_vars]

        # get rewards from the environment and evaluate rewarded gradients
        #  and feed it to agent and then call train operation
        self.rewarded_grads_placeholders_list = []
        rewarded_grads_and_vars = []
        for grad, var in grads_vars:
            rewarded_grad_placeholder = tf.placeholder(dtype=tf.float32, shape=grad.shape)
            self.rewarded_grads_placeholders_list.append(rewarded_grad_placeholder)
            rewarded_grads_and_vars.append((rewarded_grad_placeholder, var))

        self.train_operation = self.optimizer.apply_gradients(rewarded_grads_and_vars)

        self.saver = tf.train.Saver()

        config = tf.ConfigProto(
            device_count={'GPU': 0}
        )

        self.ses = tf.Session(config=config)
        self.ses.run(tf.global_variables_initializer())

    def get_action_and_gradients(self, obs):
        action, gradients = self.ses.run([self.action, self.gradients], feed_dict={self.input: obs})
        # compute network's action and gradients given the observations
        return action, gradients

    def train(self, rewarded_gradients):
        feed_dict = {}
        for i in range(len(rewarded_gradients)):
            feed_dict[self.rewarded_grads_placeholders_list[i]] = rewarded_gradients[i]
            # feed_dict[self.rewarded_grads_placeholders_list[i]] = self.ses.run(rewarded_gradients[i])
        self.ses.run(self.train_operation, feed_dict=feed_dict)
        # feed gradients into the placeholder and call train operation

    def save(self):
        self.saver.save(self.ses, "SavedModel/")

    def load(self):
        self.saver.restore(self.ses, "SavedModel/")


epochs = 100
max_steps_per_game = 1000
games_per_epoch = 100
discount_factor = 0.99
learning_rate = 0.01
all_rewards = []
agent = Agent(learning_rate)
game = gym.make("CartPole-v0").env


def geometric_progression(alpha, n):
    return (1 - alpha ** n) / (1 - alpha)


for epoch in range(epochs):
    epoch_rewards = []
    epoch_gradients = []
    epoch_average_reward = 0
    for episode in range(games_per_epoch):
        obs = game.reset()
        step = 0
        single_episode_rewards = []
        single_episode_gradients = []
        game_over = False
        while not game_over and step < max_steps_per_game:
            step += 1
            # image = game.render(mode='rgb_array') # Call this to render game and show visual
            action, gradients = agent.get_action_and_gradients(obs.reshape(-1, 4))
            action = action[0, 0]
            obs, reward, game_over, info = game.step(action)
            single_episode_rewards.append(reward)
            single_episode_gradients.append(gradients)

        epoch_rewards.append(single_episode_rewards)
        epoch_gradients.append(single_episode_gradients)
        epoch_average_reward += sum(single_episode_rewards)

    epoch_average_reward /= games_per_epoch
    all_rewards.append(epoch_average_reward)
    print("Epoch = {}, Average reward = {}".format(epoch, epoch_average_reward))

    baseline = geometric_progression(discount_factor, epoch_average_reward)
    normalized_rewards = [geometric_progression(discount_factor, len(epoch_rewards[i])) - baseline for i in
                          range(games_per_epoch)]

    mean_rewarded_gradients = [np.zeros(shape=i.shape) for i in agent.gradients]

    for i, episode_gradient in enumerate(epoch_gradients):
        for gradient_list in episode_gradient:
            for j, gradient_tensor in enumerate(gradient_list):
                mean_rewarded_gradients[j] += gradient_tensor * normalized_rewards[i]

    # if epoch_average_reward > 850:
    # print("save?")
    # if input() == "y":
    #     agent.save()
    #     print("saved")
    # print("before")
    agent.train(mean_rewarded_gradients)
    # print("after")

    # print("stop or continue? (s/c)")
    # if input() == "s":
    #     break

agent.save()
game.close()
print(all_rewards)

# Run this part after training the network
# agent = Agent(0)
# game = gym.make("CartPole-v0").env
# agent.load()
# score = 0
# for i in range(10):
#     obs = game.reset()
#     game_over = False
#     while not game_over:
#         score += 1
#         image = game.render(mode='rgb_array')  # Call this to render game and show visual
#         action, _ = agent.get_action_and_gradients(obs.reshape(-1, 4))
#         obs, reward, game_over, info = game.step(action[0,0])
#     # print(score)
#
# print("Average Score = ", score / 10)
