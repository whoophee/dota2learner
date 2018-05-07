from pybloom_live import ScalableBloomFilter
import requests
import tensorflow as tf

class D2Wrapper:
    def query_api(self, method, params):
        params['key'] = self.steam_api_key
        url = "https://api.steampowered.com/IDOTA2Match_570/{}/V001/".format(method)
        response = requests.get(url, params = params)
        ret = response.json()
        return ret.get('result', {})

    def get_match_history(self, params):
        return self.query_api("GetMatchHistory", params)

    def get_match_details(self, params):
        return self.query_api("GetMatchDetails", params)

    def __init__(self, steam_api_key):
        self.steam_api_key = steam_api_key

class CustomSBF(ScalableBloomFilter):
    def add_many(self, items):
        for item in items:
            self.add(item)

    def get_non_members(self, items):
        return [item for item in items if item not in self]

class NNModel:
    def fit(self, X, Y):
        print("Training started")
        self.op_cache = tf.identity(self.output_layer)
        self.sess.run([self.optimizer, self.cost], feed_dict = {self.x: X, self.y: Y})
        print("Training finished")

    def close(self):
        self.sess.close()
    def __init__(self, neurons, learning_rate = 0.01, seed = 128):
        # define placeholders
        self.x = tf.placeholder(tf.float32, [None, neurons[0]])
        self.y = tf.placeholder(tf.float32, [None, neurons[-1]])
        weights = []
        biases = []
        layers = []
        for i in range(1, len(neurons) - 1):
            weights.append(tf.Variable(tf.random_normal([neurons[i], neurons[i + 1]], seed=seed)))
            biases.append(tf.Variable(tf.random_normal([neurons[i + 1]], seed=seed)))

        # Build the network
        prev = self.x
        for i in range(len(weights)):
            tmp = tf.add(tf.matmul(prev, weights[i]), biases[i])
            tmp = tf.nn.relu(tmp)
            prev = tmp

        self.output_layer = tf.clip_by_value(prev, 1e-10, 0.9999999)

        self.cost = -tf.reduce_mean(tf.reduce_sum(self.y * tf.log(self.output_layer) + (1 - self.y) * tf.log(1 - self.output_layer), axis=1))
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(self.cost)
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
