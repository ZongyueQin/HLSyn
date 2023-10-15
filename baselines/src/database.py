'''

05/29/2023: A database object to use either the redis database or a custom object.

'''

from config import FLAGS
from saver import saver
from collections import OrderedDict

class OurDatabase(object):
    def __int__(self):
        self.flushdb()

    def flushdb(self):
        self.d = {}
        self.d_decoded = {}

    def hmset(self, n, data):
        self.d[n] = data
        new_data = OrderedDict()
        assert type(data) is dict
        for k, v in data.items():
            new_data[k.decode('utf-8')] = v
        self.d_decoded[n] = new_data

    def hkeys(self, n):
        if n not in self.d:
            return []
        return list(self.d[n].keys())

    def hget(self, n, key):
        try:
            return self.d_decoded[n][key]
        except KeyError as e:
            exit()


def create_database():
    if FLAGS.use_redis:
        import redis
        rtn = redis.StrictRedis(host='localhost', port=6379)
        saver.log_info(f'Connection established to port 6379')
        return rtn
    else:
        return OurDatabase()
