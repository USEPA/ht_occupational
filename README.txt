## High-throughput occupational exposure modeling ##

Code to support: Minucci et al. 2023. Data-Driven Approach to Estimating Occupational Inhalation Exposure Using Workplace Compliance Data. Environmental Science and Technology. https://doi.org/10.1021/acs.est.2c08234

prerequisites: Anaconda or miniconda previously installed (Python 3.8 or higher)

1. In the Anaconda Prompt (Windows) or bash (Linux), navigate to the directory that this README is located in.

2. To install the required package in a conda virtual environment, run:

```
conda config --append channels conda-forge
conda create --name occupation theano m2w64-toolchain -y
conda activate occupation
pip install -r requirements.txt
```

3. Now you are ready to run the scripts. They should be run in the following order:

```
python data_processing.py

python fit_hurdle_model.py
```

If fit_hurdle_model.py is too slow on your system, you can optionally load pre-fit model objects. All of the same model performance metrics and predictions will be made:

```
python load_hurdle_model.py
```

4. Output files will appear in the `output` folder. Performance metrics will be printed in the console and also to the file `fit_hurdle_model.log`.

5. To create plots:

```
python plotting.py
```

5. To use the pre-fit model to make CDR predictions:

```
python predict_CDR.py
```