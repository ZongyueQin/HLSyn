## Run

To run the training and inference processes, modify `config.py` and run `python main.py`. A log folder will be created under `src/logs` with hyperparameter details which are specified by `config.py`.


## Tips

1. You may need to set `force_regen` to True in `config.py` in order to pre-process the raw data. Once the preprocessed datasets are generated and stored as pickle files under `save`, you can set  `force_regen` to False for experiments. If some reason the datasets need to be re-regenerated, simply set `force_regen` to True, and the code will delete any existing pickle files and re-generate/pre-process the dataset. 

It may seem a bit confusing at this point, but we believe this parameter can be useful in practice, e.g. when you want to introduce some new features to the dataset, and thus instead of manually deleting the earlier pre-processed data files, you can just set this parameter to True.

In the released hyperparameter files under `configs_for_reproducibility`, some files set `force_regen` to True but some set to False -- It should always be set to True if you run the training/inference process for the first time on that machine. 

2. When training, the `subtask` parameter in `config.py` should be set to "train". When testing (with adaptation to the six held-out kernels), set it to  "inference". In addition, specify the trained model path `load_model`, e.g. "train_2023-05-30T22-10-35.899225_class_<server_name>/val_model_state_dict.pth". The code will automatically load that checkpoint.

If you encounter errors during inference, it is very likely due to mismatched hyperparameters and suggest you need to modify `config.py`'s hyperparameters to match the ones you used to train that model. For example, the model "train_2023-05-30T22-10-35.899225_class_<server_name>/val_model_state_dict.pth" corresponds to a transformer-based model that receives the source code text as input. If `config.py` sets `multi_modality` to True, meaning you want to have a model receiving both the source code text and the assembly code graph as inputs, then there might be an error when the code tries to load that checkpoint. In such cases, manually set `multi_modality` to False along with other parameters in `config.py` to the ones you used to train that model will fix the issues. 

In short, when loading a trained model's checkpoint, please double-check the `config.py` file to ensure the hyperparameters you specify match the ones used to train that model.

4. If `no localdse error` when pickle loading objects, try marking `dse_database` as source directory and let PyCharm handle the addition of it to PYTHONPATH. 

## Data Files

Download from https://zenodo.org/record/8034115 and extract under the project root directory with the folder name being "dse_database<version>". Please change the "dse_database_name" parameter in `config.py` to reflect the actual dse folder name, e.g. `dse_database_06122023`. The reason is that, we may release newer version of our dataset in future, and in that case, a different folder such as `dse_database_06122023_new` can be added and the only change to code is the "dse_database_name" parameter in `config.py`. In other words, we maintain different `dse_database`s under the project directories each corresponding to one version of our dataset. 



