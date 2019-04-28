import os
import time
from . import logger
from enum import Enum


class Stats:
    class KeyType(Enum):
        SCALAR = 'scalar'
        LIST = 'list'

        class InvalidKeyTypeException(Exception):
            pass

    def __init__(self, name):
        self.name = name
        self.stats_dict = {}
        self.stats_dict['key_types'] = {}

    def declare_key(self, key, key_type):
        assert key not in self.stats_dict.keys()
        self.set_key_type(key, key_type)
        if key_type == Stats.KeyType.SCALAR:
            self.stats_dict[key] = None
        elif key_type == Stats.KeyType.LIST:
            self.stats_dict[key] = []
        else:
            raise Stats.KeyType.InvalidKeyTypeException("Unknown key type {0}".format(key_type))

    def get_key_type(self, key, d=None):
        return self.stats_dict['key_types'].get(key, d)

    def set_key_type(self, key, key_type):
        self.stats_dict['key_types'][key] = key_type

    def get(self, key, default=None):
        if key not in self.stats_dict:
            return default
        else:
            return self.stats_dict[key]

    def record(self, key, value):
        if key not in self.stats_dict:
            self.declare_key(key, Stats.KeyType.SCALAR)
        assert self.get_key_type(key) == Stats.KeyType.SCALAR
        self.stats_dict[key] = value

    def record_append(self, key, value):
        if key not in self.stats_dict:
            self.declare_key(key, Stats.KeyType.LIST)
        assert self.get_key_type(key) == Stats.KeyType.LIST
        self.stats_dict[key].append(value)

    def record_start_time(self):
        '''A convinience method for `record('start_time', time.time())`'''
        self.record('start_time', time.time())

    def record_time_append(self, key):
        '''A convinience method for `stats_recorder.record_append(key, time.time() - stats_recorder.get('start_time')`'''
        self.record_append(key, time.time() - self.get('start_time'))

    def record_time(self, key):
        '''A convinience method for `stats_recorder.record(key, time.time() - stats_recorder.get('start_time')`'''
        self.record(key, time.time() - self.get('start_time'))

    def save(self, path=None):
        if path is None:
            path = os.path.join(logger.get_dir(), '{0}.json'.format(self.name))
        with open(path, 'w') as f:
            f.writelines(str(self.stats_dict))


stats = Stats('stats')
