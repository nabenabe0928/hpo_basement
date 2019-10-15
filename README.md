# The basement for the experiments for hyperparameter optimization (HPO)

## Requirements
・python3.7

・ConfigSpace[ (github)](https://github.com/automl/ConfigSpace)

## Implementation
An easy example of `main.py`.

```
from utils import HyperparameterUtilities
from optimizer import NelderMead

if __name__=='__main__':
    hpu = HyperparameterUtilities(
          "Sphere", # the name of objective function
          "NelderMead", # the name of an optimizer
          0, # the index number of experiments
          ["loss", "acc"], # the name of performance measurements (1st one is the main measurement.)
          dim=10 # the dimension of input (required only when the objective function is benchmark function.)
          )
    opt = NelderMead(
          hpu,
          n_parallels=1, # the number of parallel resources
          n_init=10, # the number of initial samplings
          max_evals=100 # the number of evaluations in an experiment
          )
    opt.optimize()
```

Run from termianl by `python main.py`.

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

### 1. First key

The name of objective function and it corresponds to the name of objective function class.

### 2. func_dir

the name of directory containing the objective function's class file.

### 3. config

The information related to the hyperparameters.

#### 3-1. the name of each hyperparameter

used when recording the hyperparameter configurations.

#### 3-2. lower, upper

The lower and upper bound of the hyperparameter.
Required only for float and integer parameters.

#### 3-3. dist (required anytime)

The distribution of the hyperparameter.
Either 'uniform' or 'cat'.

#### 3-4. q

The quantization parameter of a hyperparameter.
If omited, q is going to be None.
Either any float or integer value or 'None'.

#### 3-5. log

If searching on a log-scale space or not.
If 'True', on a log scale.
If omited or 'False', on a linear scale.

#### 3-6. var_type (required anytime)

The type of a hyperparameter.
Either 'int' or 'float' or 'str' or 'bool'.

#### 3-7. choices (required only if dist is 'cat')

The choices of categorical parameters.
Have to be given by a list.

An example follows below.

```
{
    "Sphere": {
      "func_dir": "benchmarks",
      "config": {
            "x": {
                "lower": -5.0, "upper": 5.0,
                "dist": "uniform", "var_type": "float"
            }
        }
    },
    "CNN": {
      "func_dir": "ml",
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
