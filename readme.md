

### Training

#### Install requirements

We have used the kmeans written in PyTorch, and the code can be found
at https://github.com/Hzzone/torch_clustering.

We need first clone it:

```shell
git clone --depth 1 https://github.com/Hzzone/torch_clustering tmp && mv tmp/torch_clustering . && rm -rf tmp
```

and then install other requirements:

```shell
pip install -r requirements.txt
```

#### Download the datasets

The datasets used in the Proposed Algorithm can be downloaded from official websites. The datasets cifar10, cifar100 and stl10 are automatically downloaded and saved in the /Dataset folder. For other datasets, you need to provide the folder destination on --data_folder input.

#### Training Commands
The config files are in `experiment/BYOL_s/cifar_10`, just run the following command:
```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3 # use the first 4 GPUs
torchrun --master_port 17673 --nproc_per_node=4 main.py experiment/BYOL_s/cifar_10/cifar10_r18_byol_s.yml
```
or
```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3 # use the first 4 GPUs
python -m torch.distributed.launch --nproc_per_node=4 --master_port=17672 main.py experiment/BYOL_s/cifar_10/cifar10_r18_byol_s.yml
```

we can also run the proposed model for n times sequentially using the following command:
```shell
cd experiment/BYOL_s/cifar_10/
bash run_n_times.sh
```

We can also enable the WANDB to visualize the training!

Set the `wandb` parameters to true, and login to wandb.ai:
```shell
wandb login xxx
```



