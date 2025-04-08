#!/bin/bash

echo "Explore..."
Rscript explore.R

echo "Wrangle..."
Rscript wrangle.R

echo "Train baselearners..."
python train_baselearner.py means
python train_baselearner.py medians
python train_baselearner.py sds
python train_baselearner.py rmssds

echo "Executing train_metalearner..."
python train_metalearner.py

echo "Executing evaluate..."
python evaluate.py
