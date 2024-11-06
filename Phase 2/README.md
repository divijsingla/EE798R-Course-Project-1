# How to Run?
1. Create a conda environment with python version = 3.7
2. Install requirements as listed in requirements.txt
3. Go to train_LFW.ipynb
4. Run each cell one-by-one to get results
5. Other files like casia_dataset_analysis.ipynb and train_caisa.ipynb have some other results or insights which are mentioned in the report. We can run those files cell-by-cell to get those results again.

IMPORTANT: The results after running the code on GPU are stored in .npy files which can be found in Divij folder in the remote host given to us. The files were too large (around 7GB) to be uploaded here. Running few blocks takes hours of time even on GPU, hence we can use those files to save time.

# How to run on remote host?
1. Go to Divij folder
2. Run conda activate divij
3. Now you can run train_LFW.ipynb file to get results as .npy files are there in that directory and loaded in the code directly.

