![MIP Workshop 2023](mip-2023-logo.png "MIP Workshop 2023")
# MIPcc23: The MIP Workshop 2023 Computational Competition

## <a id="Instructions"></a>Instructions 

### Activate conda env

Our mipcomp.sh file can be called from the submission directory.   
First, as conda environment is used for python dependencies management, the command :
```python
conda activate bb4
```


muste be run. If bb4 is not accessible, it can be recreated with the environment.yml file : 
```python
conda env create -f environment.yml
```

### Evaluation
Then, to evaluate for instance the script on obj 2 series, one could run :

```python
sh mipcomp.sh home/paul/MIPcc23/datasets/testfiles/obj_series_2.test
```

Of course, any other absolute path to a ``series.test`` file can be provided.   
Please, feel free to write to me (strangpaul21@outlook.fr) if you are having trouble with running the script for the evaluation !




