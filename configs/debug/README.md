# debug

Find below the existing configs for the debug folder. To have your YAML file indexed, add a docstring at the beginning of the file.
The docstring should be a series of comment lines starting with two '#' characters.

Example:
```
## This is a docstring
## describing the YAML file.
key: value
```


## fdr.yaml

runs 1 train, 1 validation and 1 test step


## limit.yaml

uses only 1% of the training data and 5% of validation/test data


## overfit.yaml

overfits to few batches


## profiler.yaml

runs with execution time profiling
