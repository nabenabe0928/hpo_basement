from typing import Callable, Dict, Any, Union, List
import ConfigSpace.hyperparameters as CSH


ObjectiveFuncType = Callable[[Dict[str, Any], int, str], Dict[str, Union[float, int]]]
HyperparameterTypes = Union[CSH.CategoricalHyperparameter,
                            CSH.UniformFloatHyperparameter,
                            CSH.UniformIntegerHyperparameter]
ConfigurationTypes = Union[Dict[str, Union[float, int, str]],
                           List[Union[float, int, str]]]
