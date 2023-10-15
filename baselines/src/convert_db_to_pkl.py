'''

05/29/2023: Convert db files to pickle files so that no need to set up redis on server.

'''
from utils import get_root_path, save_pickle
from glob import glob
from tqdm import tqdm
from os.path import basename, dirname
import redis, pickle
from collections import OrderedDict

db_files = glob(f'/{get_root_path()}/dse_database/**/*.db', recursive=True)
database = redis.StrictRedis(host='localhost', port=6379)
database.flushdb()

for db_file in tqdm(db_files):
    print(db_file)
    bn = basename(db_file)
    assert '.db' in bn
    fnb = bn.split('.db')[0]
    new_file = f'{dirname(db_file)}/{fnb}.pkl'
    print(new_file)

    database.flushdb()

    d = OrderedDict()
    try:
        with open(db_file, 'rb') as f_db:
            database.hmset(0, pickle.load(f_db))
        keys = [k.decode('utf-8') for k in database.hkeys(0)]
        print(len(keys))
        for key in sorted(keys):
            # obj = pickle.loads(database.hget(0, key))
            d[key] = database.hget(0, key)
            # print()
        save_pickle(d, new_file, print_msg=True)
    except Exception as e:
        print(e)

