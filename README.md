To reproduce this work, unpack the ZIP file and save to a folder path, navigate to the root directory, and download/activate the environment by the following commands: 
```
conda env create -f environment.yml
conda activate seminar
```

Note: An anaconda distribution is required to execute these commands. Also, this project was completed on a Windows operating system, so the compiled Anaconda environment may cause issues for users on other operating systems. If any packages or dependencies fail to download, simply install the package versions specified in `environment.yml` on your local machine. 

Once the conda environment has been activated, results can be reproduced by navigating to the scripts folder and executing `pipeline.sh`. The only plot that is not rendered by this Bash script is the SHAP plot (see `get_shaps.ipynb`). Please note that execution of this code may take a while to run, but no more than an hour. Also, please note that you may need to make `pipeline.sh` executable, which is shown in the second line below:
```
cd scripts
chmod +x pipeline.sh 
./pipeline.sh
```

Lastly, please find the details of what each folder contains below: 
1. `data` - raw data and instructions provided by Vertex team
2. `objects` - mutated data used during modeling
3. `ensemble` - transformed train/test data used during modeling
4. `results` - tables and figures found in presentation
5. `scripts` - all scripts used to produce everything outside of `data`
    - `explore.R` - executes code that reflects information in the "explore" section of the presentation
    - `wrangle.R` - wrangles the data as depicted in the "wrangle" section of the presentation
    - `utils.R` - contains helper functions used in R scripts
    - `train_baselearner.py` - trains base learners as depicted in the "train" section of the presentation; expects a command-line argument that specifies the time series transformation of interest
    - `train_metalearner.py` - trains meta learners as depicted in the "train" section of the presentation
    - `evaluate.py` - produces the results found in the "evaluate" section of the presentation
    - `get_shaps.ipynb` - extracts SHAP values from meta learner; NOTE: run outside of the conda environment locally, as the shap package conflicted with other packages required to execute `./pipeline.sh`
