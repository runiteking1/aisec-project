# AISEC Project

This repository contains the code and documentation for the final AISEC Project.
The goal of the project is to investigate adversarial robustness and sharpness of loss landscapes of higher-order optimizer, specifically Gauss-Newton compared to Adam. 

# Structure

aisec-project/

├── conf/ # Configuration files (e.g. experiments, hyperparameters)

├── doc/ 

│ └── paper.pdf # Small draft detailing experiments

├── src/ # Source code: models, training, evaluation, utilities

│ └── multirun/ # Sweeps over models/rho

│ └── outputs/ # Run logs from my runs. 

├── requirements.txt # All installed dependencies

└── .gitignore

Note that model parameters are uploaded into the Hydra run folders and should be easily replicated. 

