from utils import *
import configparser
from pymongo import MongoClient
import numpy as np
import time
from keras.models import Sequential
from keras.layers import Dense
from threading import Thread, Lock
import tensorflow

PUBLIC_MATCHMAKING = 0
PRACTISE = 1
TOURNAMENT = 2
TUTORIAL = 3
COOP_WITH_BOTS = 4
TEAM_MATCH = 5
SOLO_QUEUE = 6
RANKED_MATCHMAKING = 7
SOLO_MID = 8
NS = 1
HS = 2
VHS = 3

get_val = lambda x: -1 if x == 1 else 1
player_slot_val = {x:get_val(int(x / 128)) for x in [0, 1, 2, 3, 4, 128, 129, 130, 131, 132]}

def generate_set(matches):
    X = []
    Y = []
    for m in matches:
        tmp = [0]*121
        for player in m.get('players'):
            tmp[player['hero_id']] = player_slot_val[player['player_slot']]
        X.append(tmp)
        Y.append(m['radiant_win'])
    X = np.array(X)
    Y = np.array(Y)
    X[Y == 0] = -X[Y == 0]
    Y = X
    Y[Y == -1] = 0
    return X, Y

class D2Trainer:

    def match_queue(self, lobby, skill):

        while not self.killed:
            response = self.d2wrapper.get_match_history({'skill' : skill})
            # Build list of latest ranked games.
            latest_matchids = [match['match_id'] for match in response.get('matches', []) if match['lobby_type'] == lobby]
            # Get unused data points from bloom filter
            unused_matchids = self.sbf.get_non_members(latest_matchids)
            tmp = []
            for matchid in unused_matchids:
                game_result = self.d2wrapper.get_match_details({'match_id':matchid})
                tmp.append(game_result)

            self.lock.acquire()
            self.match_queue.extend(tmp)
            self.lock.release()

            self.sbf.add_many(unused_matchids)

            time.sleep(5)

    def train_nn(self, batch_size = 20):


        i = 0
        while not self.killed:
            if len(self.match_queue) < batch_size:
                time.sleep(30)
                continue
            i += batch_size
            self.lock.acquire()
            tmp = self.match_queue[:batch_size]
            self.match_queue = self.match_queue[batch_size:]
            self.lock.release()

            X, Y = generate_set(tmp)
            self.nn_model.fit(X, Y)
            print("Trained with {} samples so far.".format(i))
        self.nn_model.close()

    def start(self):
        if self.killed:
            self.match_queue = []
            if self.mongodb_uri:
                self.client = MongoClient(self.mongodb_uri)
            self.killed = False
            self.gq_thread.start()
            self.nn_thread.start()


    def stop(self):
        if not self.killed:
            self.killed = True
            print("Killing game queue thread.")
            self.gq_thread.join()
            print("Killing neural net thread.")
            self.nn_thread.join()
            print("Threads exited gracefully.")
            self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()

    def __init__(self, api_key, mongodb_uri = None, lobby_type = RANKED_MATCHMAKING, skill = VHS):
        self.nn_model = NNModel((121, 121, 121))
        self.killed = True
        self.lock = Lock()
        self.gq_thread = Thread(target = self.match_queue, args = (lobby_type, skill))
        self.nn_thread = Thread(target = self.train_nn)
        self.mongodb_uri = mongodb_uri
        self.sbf = CustomSBF(mode = ScalableBloomFilter.LARGE_SET_GROWTH)
        self.d2wrapper = D2Wrapper(api_key)


if __name__ == "__main__":
    import configparser
    config = configparser.ConfigParser()
    config.read('default.ini')
    cur_config = config['default']
    with D2Trainer(cur_config['steam_api_key'], cur_config['mongodb_uri']) as test_model:
        test_model.start()
        while True:
            try:
                time.sleep(1)
            except KeyboardInterrupt:
                break
