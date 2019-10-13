# The basement for the experiments for hyperparameter optimization (HPO)

## Requirements
・python3.7

・ConfigSpace[ (github)](https://github.com/automl/ConfigSpace)

## Optimizer
You can add whatever optimizers you would like to use in this basement.
By inheriting the `BaseOptimizer` object, you can use basic function needed to start HPO. 
A small example follows below:

```
from optimizer.base_optimizer import BaseOptimizer
import utils

class OptName(BaseOptimizer):
    def __init__(self,
                 hpu, # hyperparameter utility object 
                 n_parallels=1, # the number of parallel computer resourses
                 n_init=10, # the number of initial sampling
                 max_evals=100, # the number of maximum evaluations in an experiment
                 **kwargs
                 ):

        # inheritance (if rs is True, Random Search)
        super().__init__(hpu, rs=False, n_parallels=n_parallels, n_init=n_init, max_evals=max_evals)

        # optimizer in BaseOptimizer object
        self.opt = self.sample
    
    def sample(self):
        """
        some procedures and finally returns a hyperparameter configuration
        this hyperparameter configuration must be on usual scales. 
        """

        return hp_conf
```

## Hyperparameters of Objective Functions
Describe the details of hyperparameters in `params.json`.

・First key

The name of objective function and it corresponds to the name of objective function class. 

・func_dir

the name of directory containing the objective function's class file.

・dim

The dimension of the hyperparameters of the objective function.

・config

The information related to the hyperparameters.

ーthe name of each hyperparameter

used when recording the hyperparameter configurations.

ーlower, upper

The lower and upper bound of the hyperparameter. 
Required only for float and integer parameters.

ーdist (required anytime)

The distribution of the hyperparameter. 
Either 'uniform' or 'cat'.

ーq

The quantization parameter of a hyperparameter. 
If omited, q is going to be None. 
Either any float or integer value or 'None'.

ーlog

If searching on a log-scale space or not.
If 'True', on a log scale.
If omited or 'False', on a linear scale. 

ーvar_type (required anytime)

The type of a hyperparameter.
Either 'int' or 'float' or 'str' or 'bool'.

ーchoices (required only if dist is 'cat')

The choices of categorical parameters.
Have to be given by a list.

An example follows below.

```
{
    "Sphere": {
      "func_dir": "benchmarks", "dim": 5, 
      "config": {
            "x": {
                "lower": -5.0, "upper": 5.0,
                "dist": "uniform", "var_type": "float"
            }
        }
    },
    "CNN": {
      "func_dir": "ml", "dim": 4, 
      "config": {
            "batch_size": {
                "lower": 32, "upper": 256,
                "dist": "uniform", "log": "True",
                "var_type": "int"
            },
            "lr": {
                "lower": 5.0e-3, "upper": 5.0e-1,
                "dist": "uniform", "log": "True",
                "var_type": "float"
            },
            "momentum": {
                "lower": 0.8, "upper": 1.0,
                "dist": "uniform", "q": 0.1,
                "log": "False", "var_type": "float"
            },
            "nesterov": {
                "dist": "cat", "choices": [True, False],
                "var_type": "bool"
            }
        }
    }
}
  
```

## Objective Functions

The target objective function in an experiment.
This function must receive the `n_gpu` and `hp_conf` from `BaseOptimizer` object and return the performance by a dictionary format.
An example follows below.


```

"""
Parameters
----------
hp_conf: 1d list of hyperparameter value
    [the index for a hyperparameter]
n_gpu: int
    the index of a visible GPU

Returns
-------
ys: dict
    keys are the name of performance measurements.
    values are the corresponding performance.
"""

def Sphere(hp_conf, n_gpu=None):
    return {"loss": (np.array(hp_conf) ** 2).sum()}
```