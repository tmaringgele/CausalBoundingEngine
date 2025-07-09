from causalboundingengine.scenario import Scenario
from causalboundingengine.algorithms.tianpearl import TianPearl
from causalboundingengine.algorithms.manski import Manski
from causalboundingengine.algorithms.entropybounds import Entropybounds
from causalboundingengine.algorithms.zhangbareinboim import ZhangBareinboim



#Collection of scenarios for the Causal Bounding Engine

# Binary Confounding
class BinaryConf(Scenario):
    AVAILABLE_ALGORITHMS = {
        'ATE': {
            'manski': Manski,
            'tianpearl': TianPearl,
            'entropybounds': Entropybounds
        },
        'PNS': {
            'tianpearl': TianPearl,
            'entropybounds': Entropybounds
        }
    }

# Binary Instrumental Variable
class BinIV(Scenario):
    AVAILABLE_ALGORITHMS = {
        'ATE': {
            'manski': Manski,
            'tianpearl': TianPearl,
            'entropybounds': Entropybounds
        },
        'PNS': {
            'tianpearl': TianPearl,
            'entropybounds': Entropybounds
        }
    }


# Continuous Instrumental Variable
class ContIV(Scenario):
    AVAILABLE_ALGORITHMS = {
        'ATE': {
            'zhangbareinboim': ZhangBareinboim
        }
    }