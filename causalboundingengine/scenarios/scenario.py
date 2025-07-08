from abc import ABC, abstractmethod
import time
from datetime import datetime
from causalboundingengine.utils.data import Data

class Scenario:
    AVAILABLE_ALGORITHMS = {}  # to be defined in each subclass

    def __init__(self, X, Y, Z=None):
        self.data = Data(X, Y, Z)
        self.ATE = AlgorithmDispatcher(self, 'ATE')
        self.PNS = AlgorithmDispatcher(self, 'PNS')

    def get_algorithms(self, query_type):
        return list(self.AVAILABLE_ALGORITHMS.get(query_type, {}).keys())


class AlgorithmDispatcher:
    def __init__(self, scenario, query_type):
        self.scenario = scenario
        self.query_type = query_type

    def __getattr__(self, name):
        def _wrapped(*args, **kwargs):
            cls = self.scenario.AVAILABLE_ALGORITHMS[self.query_type][name]()
            return cls.bound_ATE(**self.scenario.data.unpack(), *args, **kwargs)
        return _wrapped
