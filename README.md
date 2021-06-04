# README

## Intro
`pyspark_helper.py` is a Python module that consists of some of my commonly executed transformation defined as functions. There is a Jupyter notebook `example.ipynb` that demonstrates the functionalities of `pyspark_helper.py`.

## Using venv in Jupyter notebook
- Include the following lines in the notebook
```
import os

os.environ['PYSPARK_PYTHON'] = "./venv/bin/python"
```

## Build egg distribution file
1. Do pip install -r requirements.txt 
2. python setup.py bdist_egg
> This will create a .egg file in dist/ directory 
> 
> This .egg file can be submitted to spark using the `spark.submit.pyFiles` option. For example
```
conf = spark.sparkContext._conf.setAll(
    [
	...
        ('spark.submit.pyFiles','dist/pyspark_helper-0.1.0-py3.6.egg'),
    ]

spark = SparkSession.builder.config(conf=conf).getOrCreate()
```
