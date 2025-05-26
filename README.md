# LaDEEP (KDD 2025)

Code for "LaDEEP: A Deep Learning-based Surrogate Model for Large Deformation of Elastic-Plastic Solids", accepted at KDD 2025 ADS Track.

## Usage

We recommend that `python=3.10`.

### Install packages

```
pip install -r ./requirements.txt
```

### Dataset

You can download the dataset from the [link](https://drive.google.com/drive/folders/1DJCi74Gg3d4cBv1JlUG1pmyS1pVo2mJm?usp=sharing).

After obtaining the dataset, put it in the `./data` folder. You can also change the data folder whatever you want in `./config.ini` file.

### Train

Several hyperparameters are recorded in `./config.ini` file. You can change them to train the model. The default settings are corresponding to those used in the paper.

You need first change the `mode` in `./config.ini` into "train", and then use the below command to start training the model:

```
# mkdir ./checkpoints
# mkdir ./logs

# Display training process on the frontend
python main.py

# Display training process on the backend
nohup python main.py >> ./train.log 2>&1 &
```

The training and evaluation losses are illustrated by `tensorboard`:

```
tensorboard --logdir ./logs --port 8888
```

Then you can monitor the training and evaluation details by open `localhost:8888` on your browser.

### Test

After finishing training, you need change the `mode` in `./config.ini` into "test", and then use the below command to start testing the model:

```
python main.py

# or
nohup python main.py >> ./test.log 2>&1 &
```

The result, MAD and TE will be save in `./data/prediction_results/test_{mode_id}`. 

We have also provided the trained model weight in `./checkpoints/train_0` folder. You can directly test the model with it.


## Contact

Any further questions, please contact shilongtao@stu.pku.edu.cn.

## Citation

If you find this repo useful for you, please consider citing the following paper:
```
# TODO
```

