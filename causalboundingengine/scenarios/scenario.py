
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
            method = getattr(cls, f'bound_{self.query_type}')
            
            # Combine unpacked data and user kwargs
            combined_kwargs = {**self.scenario.data.unpack(), **kwargs}
            
            return method(*args, **combined_kwargs)
        return _wrapped
