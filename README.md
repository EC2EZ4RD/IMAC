# NIPS #3584 Source Code

This is the code for implementing of NIPS #3584 paper.

## Installation


```
conda create -n imac python=3.6
conda activate imac
pip install tensorflow==1.12.0
conda install mkl_fft=1.0.10
pip install -r requirements.txt
```

- Known dependencies: Python (3.6.8), OpenAI gym (0.9.4), tensorflow (1.12.0), numpy (1.16.2)

## How to run

To run the code, `cd` into the `experiments` directory and run `train.py`:

``
python train.py --scenario simple_spread --exp-name debug --save-dir ./result_test/debug --batch-size 1024 --ibmac_com --trainer ibmac
``

You can use tensorboard to visualize the results.

## Command-line options

### Environment options

- `--scenario`: defines which environment in the MPE is to be used (default: `"simple_spread"`)

- `--max-episode-len` maximum length of each episode for the environment (default: `25`)

- `--num-episodes` total number of training episodes (default: `60000`)

- `--num-adversaries`: number of adversaries in the environment (default: `0`)

- `--good-policy`: algorithm used for the 'good' (non adversary) policies in the environment
(default: `"maddpg"`; options: {`"maddpg"`, `"ddpg"`})

- `--adv-policy`: algorithm used for the adversary policies in the environment
(default: `"maddpg"`; options: {`"maddpg"`, `"ddpg"`})

### Core training parameters

- `--trainer`: different algorithms (default: `"imbac"`)

    `ibmac`: for training scheduler

    `ibmac_inter`: for training policy and messages output

- `--lr`: learning rate (default: `1e-2`)

- `--gamma`: discount factor (default: `0.95`)

- `--batch-size`: batch size (default: `1024`)

- `--num-units`: number of units in the MLP (default: `64`)

- `--beta`: coefficient of KL loss (default: `0.05`)

- `--ibmac_com`: boolean that enable commucniation (default: `False`)

- `--random-seed`: random seed (default: `42`)

### Checkpointing

- `--exp-name`: name of the experiment, used as the file name to save all results (default: `None`)

- `--save-dir`: directory where intermediate training results and model will be saved (default: `"/tmp/policy/"`)

- `--save-rate`: model is saved every time this number of episodes has been completed (default: `1000`)

- `--load-dir`: directory where training state and model are loaded from (default: `""`)

### Evaluation

- `--restore`: restores previous training state stored in `load-dir` (or in `save-dir` if no `load-dir`
has been provided), and continues training (default: `False`)

- `--display`: displays to the screen the trained policy stored in `load-dir` (or in `save-dir` if no `load-dir`
has been provided), but does not continue training (default: `False`)

- `--benchmark`: runs benchmarking evaluations on saved policy, saves results to `benchmark-dir` folder (default: `False`)

- `--benchmark-iters`: number of iterations to run benchmarking for (default: `100000`)

- `--benchmark-dir`: directory where benchmarking data is saved (default: `"./benchmark_files/"`)

- `--plots-dir`: directory where training curves are saved (default: `"./learning_curves/"`)

### Acknowledgement

Our code is based on the version in:
<pre>
@article{lowe2017multi,
  title={Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments},
  author={Lowe, Ryan and Wu, Yi and Tamar, Aviv and Harb, Jean and Abbeel, Pieter and Mordatch, Igor},
  journal={Neural Information Processing Systems (NIPS)},
  year={2017}
}
</pre>

We slightly modify the environment on the **act_space** setting, so there are some differences on final reward output if you directly install the original version of environment.

We also add a new scenario: `simple_spread_partially_observed`. The `num_agents` can be modified for more agents.