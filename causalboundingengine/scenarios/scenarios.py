from causalboundingengine.scenarios.scenario import Scenario
from causalboundingengine.algorithms.conf.tianpearl import TianPearl
from causalboundingengine.algorithms.conf.manski import Manski



#Collection of scenarios for the Causal Bounding Engine

# Binary Confounding
class BinaryConf(Scenario):
    AVAILABLE_ALGORITHMS = {
        'ATE': {
            'manski': Manski,
            'tianpearl': TianPearl
        },
        'PNS': {
            'tianpearl': TianPearl
        }
    }
