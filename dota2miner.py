import requests
from pybloom_live import ScalableBloomFilter
from threading import Thread, Lock
import time
import argparse
import os
import configparser

def _print(*args):
    print('['+time.strftime('%X %x')+']', *args, '\n')

class CustomSBF(ScalableBloomFilter):
    def add_many(self, items):
        for item in items:
            self.add(item)

    def get_non_members(self, items):
        return [item for item in items if item not in self]

class dota2miner:
    def query_api(self, method, params):
        params['key'] = self.steam_api_key
        url = "https://api.steampowered.com/IDOTA2Match_570/{}/V001/".format(method)
        response = requests.get(url, params = params, timeout = 30)
        ret = response.json()
        return ret.get('result', {})

    def get_match_history(self, params):
        return self.query_api("GetMatchHistory", params)

    def get_match_details(self, params):
        return self.query_api("GetMatchDetails", params)

    def timed_out(self):
        if self.timeout > time.time() - self.recent_mine:
            return False
        return True

    def time_elapsed(self):
        return time.time() - self.start_time

    def _mine(self):
        self.match_queue = []
        self.start_time = time.time()
        self.recent_mine = self.start_time
        while self.running and not self.timed_out():
            response = self.get_match_history({'skill' : self.skill})
            latest_matchids = [match['match_id'] for match in response.get('matches', []) if match['lobby_type'] in self.lobby_type]

            unused_matchids = self.sbf.get_non_members(latest_matchids)
            tmp = []
            for matchid in unused_matchids:
                game_result = self.get_match_details({'match_id':matchid})
                tmp.append(game_result)

            self.lock.acquire()
            self.match_queue.extend(tmp)
            self.lock.release()
            self.sbf.add_many(unused_matchids)

            # Set most recent fetch time if matches were mined.
            if tmp:
                self.recent_mine = time.time()
            time.sleep(self.wait_time)

    def get_data(self, n = 100):
        if n > len(self.match_queue):
            return None
        self.lock.acquire()
        tmp = self.match_queue[:n]
        self.match_queue = self.match_queue[n:]
        self.lock.release()
        return tmp

    def start(self, **kwargs):
        if self.running:
            return
        self.wait_time = kwargs.get('wait_time', 5)
        self.skill = kwargs.get('skill', 0)
        self.lobby_type = kwargs.get('lobby_type', list(range(17)))
        self.timeout = kwargs.get('timeout', 60 * 20)

        self.running = True
        self.miner_thread.start()

    def stop(self):
        if not self.running:
            return
        self.running = False
        self.miner_thread.join()

    def __len__(self):
        return len(self.match_queue)

    def __repr__(self):
        ret = {'running' : self.running}
        if self.running:
            t = int(self.time_elapsed())
            h = int(t/3600)
            m = int((t % 3600)/60)
            s = int(t % 60)

            ret['Lobbies'] = self.lobby_type
            ret['Skill'] = self.skill
            ret['Games Mined'] = len(self.match_queue)
            ret['Time elapsed'] = '{}h:{}m:{}s'.format(h, m, s)
        return '\n'.join(['{} : {}'.format(key, val) for key, val in ret.items()])


    def __init__(self, steam_api_key, mongodb_uri = None):
        self.running = False
        self.steam_api_key = steam_api_key
        self.sbf = CustomSBF(mode = ScalableBloomFilter.LARGE_SET_GROWTH)
        self.miner_thread = Thread(target = self._mine, name = 'Dota2Miner')
        self.lock = Lock()

def get_configs(configfile):
    config = configparser.ConfigParser()
    config.read(configfile)
    return config

if __name__ == "__main__":
    print()

    # later allow argument
    configfile = 'default.ini'
    configsetting = 'default'
    default = get_configs(configfile)[configsetting]

    serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host = default.get('host', 'localhost')
    port = default.get('port',8089)
    serversocket.bind((host, port))
    serversocket.listen(1)
    _print('Serving at {}:{}'.format(host, port))

    miner = dota2miner(steam_api_key = default['steam_api_key'])

    miner.start()
    _print('Miner started.')


    connection, address = serversocket.accept()
    addr_str = address[0] + ':' + address[1]
    _print('{} connected.'.format(addr_str))


    while True:
        buf = connection.recv(4096)
        cmd = buf['cmd']

        if cmd == 'describe':
            _print('\n'+str(miner))

        elif cmd == 'stop':
            _print('Stopping server.')
            serversocket.close()
            break
        else:
            _print('Invalid command.')

    miner.stop()
