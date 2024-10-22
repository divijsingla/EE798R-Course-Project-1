# How to Run?
1. Run 'train.py' with correct path to CASIA-WebFace Dataset rec and idx files mentioned on lines 73, 74 respectively.
2. A file called 'dcnn_trained.params' will be created in the same directory. This will contain the learned weights.
3. Test and check the results of NMI and F-measure on CASIA-WebFace Dataset using 'validate_same.py' (give path to dataset in lines 55, 56).
4. Test and check the results of NMI and F-measure on Labelled Face in the Wild Dataset using 'validate_lfw.py' (give path to dataset in line 19).



