# PB-ML: Poisson-Boltzmann Machine Learning Model
We first trained different DNN architectures based on 367 features of 4294 PDBBind protein data, using the following 448 combinations of hyperparameters:  
epochs = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000, 1500],  
layers = ['2048,2048,512,512', '8000,8000,8000,8000', '500,500,500', '500,250,250', '12000,12000,12000', '2048,5000,8000,2048','12000,8000,8000,10000,5000','8192,8192,2048,2048,4096,4096'],  
batch_sizes = [50, 100, 200, 400],   
and No. 364 model performed the best with a minimum test MAPE (mean absolute percentage error) at 0.004023. This model is trained from Keras with batch_size = 400, epochs = 900, layers = [367,500,500,500,1], and is applied on the 195 test proteins and compared with MIBPB solver at a mesh size = 0.5. Details can be found in the paper: "Poisson-Boltzmann based machine learning (PBML) model for electrostatic analysis", https://arxiv.org/abs/2312.11482.   

In this repo, the python script "feature.py" generates the graph features, VDW and Coulomb force features, in a total number of 75.  
The python script "generate_feature.py" gathers all 367 features including the above, other protein features and GB features, as described in the paper.  
The python script "run364.py" was written for running 195 test proteins in a parallel manner. As our train and test datasets are still private, some functions and comment lines which are serving the test dataset can be neglected. The original dataset might be shared upon request. This script now has the following functions:  
it calls the softwares, i.e., the GB solver: bornRadius, the ESES solver: MS_Intersection;  
it calls the above two sciptes to prepare the features for each protein;  
it uses the trained No.364 model to predict the solvation energy difference between MIBPB at mesh size 0.2 and GB solvation energy;   
it has now been update to predict Barnase and Barstar binding simulation, the corresponding datasets are in prep_bind/data-set2;   
the X_train file containing the 367 features of 4294 proteins is here for data normalization. 
