from causalboundingengine.scenario import Scenario
from causalboundingengine.algorithms.tianpearl import TianPearl
from causalboundingengine.algorithms.manski import Manski
from causalboundingengine.algorithms.entropybounds import Entropybounds
from causalboundingengine.algorithms.zhangbareinboim import ZhangBareinboim
from causalboundingengine.algorithms.causaloptim import CausalOptim
from causalboundingengine.algorithms.zaffalonbounds import Zaffalonbounds



#Collection of scenarios for the Causal Bounding Engine

# Binary Confounding
class BinaryConf(Scenario):
    AVAILABLE_ALGORITHMS = {
        'ATE': {
            'manski': Manski,
            'tianpearl': TianPearl,
            'entropybounds': Entropybounds,
            'causaloptim': CausalOptim,
            'zaffalonbounds': Zaffalonbounds
        },
        'PNS': {
            'tianpearl': TianPearl,
            'entropybounds': Entropybounds,
            'causaloptim': CausalOptim,
            'zaffalonbounds': Zaffalonbounds
        }
    }

# Binary Instrumental Variable
class BinaryIV(Scenario):
    AVAILABLE_ALGORITHMS = {
        'ATE': {
            'causaloptim': CausalOptim,
            'zaffalonbounds': Zaffalonbounds
        },
        'PNS': {
            'causaloptim': CausalOptim,
            'zaffalonbounds': Zaffalonbounds
        }
    }


# Continuous Instrumental Variable
class ContIV(Scenario):
    AVAILABLE_ALGORITHMS = {
        'ATE': {
            'zhangbareinboim': ZhangBareinboim
        }
    }