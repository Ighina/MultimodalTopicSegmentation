# Download Dataset
In order to use the RadioNewsSBBC dataset, you first need to download it from [here](https://zenodo.org/record/7825759).

# Replicate the Results from the Paper
To replicate the results from the original paper, after having downloaded and unzipped the dataset you should have a folder named "RadioNewsSentence" inside this folder.
Once you're sure you have correctly extracted the dataset, run the following code for replicating unimodal experiments:

`./run_nonnews_unimodal.sh 10 ExperimentsUnimodal b`

Run the following for replicating early fusion experiments:

`./run_nonnews_earlyfusion.sh 10 ExperimentsEarlyFusion b`

Run the following for replicating late fusion experiments:

`./run_nonnews_latefusion.sh 10 ExperimentsLateFusion b`


The scripts will run all the experiments with all the encoders on the RadioNewsSBBC dataset and, when finished, you can find all results inside the results.txt of each subfolder
inside the newly generated Experiments folders. Each subfolder is named after the relative encoder. 
Note that the script doesn't allow two output folders with the same name and, for that, if you want to re-run the experiments change the "Experiments" argument in the
script above (e.g. Experiments2)

To generate all the p-values reported in the original paper together with all the results with different metrics, make sure of having run the training code and named the output directory "Experiments", then run the following:

`python ../calculate_accuracy_metrics_sentence.py nonnews`

A csv file named final_bilstm_results.csv will be generated inside the Experiments directories containing all results and pvalues.