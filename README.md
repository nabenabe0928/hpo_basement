# The basement for the experiments for hyperparameter optimization (HPO)

## Requirements
・python3.7

・ConfigSpace[ (github)](https://github.com/automl/ConfigSpace)

## Implementation
An easy example of `main.py`.
Note that the optimization is always minimization;
Therefore, users have to set the output multiplied by -1 when hoping to maximize.

```
import utils
import optimizer


if __name__ == '__main__':
    requirements, dim = utils.parse_requirements()
    hp_utils = utils.HyperparameterUtilities("Sphere", dim=dim)
    opt = optimizer.NelderMead(hp_utils, **requirements)
    opt.optimize()
```

Run from termianl by (one example):

```
python main.py -dim 2 -par 1 -ini 3 -exp 0 -eva 100 -res 0
```

where all the arguments are integer.

### dim
The dimension of input space.
Only for benchmark functions.

### par
The number of parallel computer resources such as GPU or CPU.

### ini
The number of initial samplings.

### exp
The index of experiments.
Used only for indexing of experiments.

### eva
The maximum number of evaluations in an experiment.
If eva = 100, 100 configurations will be evaluated.

### res
Whether restarting an experiment based on the previous experiment.

## Optimizer
You can add whatever optimizers you would like to use in this basement.
By inheriting the `BaseOptimizer` object, you can use basic function needed to start HPO.
A small example follows below:

```
from optimizer.base_optimizer import BaseOptimizer


class OptName(BaseOptimizer):
    def __init__(self,
                 hp_utils,  # hyperparameter utility object
                 n_parallels=1,  # the number of parallel computer resourses
                 n_init=10,  # the number of initial sampling
                 n_experiments=0,  # the index of experiments. Used only to specify the path of log files.
                 max_evals=100,  # the number of maximum evaluations in an experiment
                 restart=True,  # Whether restarting the previous experiment or not. If False, removes the previous log files.
                 **kwargs
                 ):

        # inheritance (if rs is True, Random Search. Default is False.)
        super().__init__(hp_utils,
                         n_parallels=n_parallels,
                         n_init=n_init,
                         n_experiments=n_experiments,
                         max_evals=max_evals,
                         restart=restart,
                         rs=False  # Whether random search or not.
                         )

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

The name of directory containing the objective function's class file.

### 3. main

The name of main function evaluating the hyperparameter configuration.

### 4. y_names

The names of the measurements of hyperparameter configurations

### 3. config

The information related to the hyperparameters.

#### 3-1. the name of each hyperparameter

Used when recording the hyperparameter configurations.

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
      "func_dir": "benchmarks", "main": "f",
      "y_names": ["loss"],
      "config": {
            "x": {
                "lower": -5.0, "upper": 5.0,
                "dist": "uniform", "var_type": "float"
            }
        }
    },
    "CNN": {
      "func_dir": "ml", "main": "train",
      "y_names": ["error", "cross_entropy"],
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
An example of (`obj_functions/benchmarks/Sphere.py`) follows below.


```
import numpy as np

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

def f(hp_conf, n_gpu=None):
    return {"loss": (np.array(hp_conf) ** 2).sum()}
```
