# FMCW-Radar-Target-Classification-ML

## Datasets
All the datasets used in this project are accessible through the OneDrive shared folder found at:
https://uweacuk-my.sharepoint.com/:f:/g/personal/siva2_bolisetti_live_uwe_ac_uk/EvrefqLAxGhDg0JP8GlrFJYBxHu0DA3aPs5Q8pcGcuuKqg?e=G0bt9d

## All custom functions are available in FMCW_Data_Simulation_Library.py

## Radar Processing Pipeline and Data Simulation
Implements the radar processing pipeline and generated the range-doppler data.
* Run FMCW_Data_Generator_MPTs-Function.ipynb with required inputs

## Check missing values
Checks the dataset for missing values.
* Run Target_Classification_ML_Doppler-Missign-Values.ipynb and specify the target directory where the required data is available

## Machine Larning Models
Trains and tests machine learning values. Optionally, the trained models can be saved to a local directory.
* Run Target_Classification_ML_Doppler_CARRADA.ipynb to train the models with CARRADA data only
* Run Target_Classification_ML_Doppler_Simulated.ipynb to train the models with simulated data only
* Run Target_Classification_ML_Doppler_Full.ipynb to train the models with the combined data
* Specify required inputs as specified in each of the scripts

## Machine Learning Model Evaluation Data
Generates evaluation data to evaluate machine learning model performance for different values of SNRs. Generated range-doppler data for each target along with target class information. Range-doppler data and target class information are saved to different files with identifiable file names. 
* Run Model_Evaluation_Data_Genarator.ipynb with required inputs.
* Data is saved to the specified local directory.

## Machine Learning Model Evaluation
Loads the evaluation data and pre-trained ML models to compare ML model performances.
* Run Model_Evaluator.ipynb with required inputs