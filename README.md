Code
We have provided partial code for the W-shape neural operator in the supplementary materials, including the implementation code for the architecture called WFNO and the W-shape FNO code after migrating our architecture to deep FNO called MFNO.If you want to run these codes in LSM environment, please modify model_dict.py accordingly

Parameter
Due to some implementation issues, we conducted comparative experiments on Darcy and Plasticity tasks in the LSM code environment, and compared Navier Stokes, Pipe, Airfoils, and Elasticity tasks in the PDEBench architecture. The architecture code we provide runs in an LSM environment, but migrating to PDEBench requires specifying some manually specified padding parameters. We have synchronized with LSM in the comparative experiment, and this part of the padding parameters can be set according to the code of the LSM model.For this reason, we have provided the corresponding parameter list while providing the configuration file. These codes will be made public after modification. Other parameters can be found in the paper or are default values. A file with .yaml suffix is a configuration file that runs on the PDEBench framework, while .sh is a configuration file that runs in the LSM code.

Datasets
In addition, in order to process the standard datasets of FNO and geo-FNO on the PDEBench framework, we need to convert the original dataset to the HDF5 format dataset. We will also provide the corresponding conversion code which are stored in the data_convert.py in the supplementary materials. And then you should write the data processing section required for data loading in utils.py of PDEBench based on the data conversion code.

Evaluation
The evaluation metrics we provide, such as fRMSE and relative fRMSE, as well as the implementation of spectrogram generation, are stored in the metric.py file which we make modifications based on PDEBench's metric.py. If you want to test these metrics during runtime, you can refer to metric.py for implementation.
