After unzip the zip file, first Go to the path of files:
```
cd hw1
```

For each section, run the following commands: 

## Behaviour Cloning
Question1.2:
For Ant environment:
```
 python rob831/scripts/run_hw1.py --expert_policy_file rob831/policies/experts/Ant.pkl --env_name Ant-v2 --exp_name bc_ant --n_iter 1 --expert_data rob831/expert_data/expert_data_Ant-v2.pkl --video_log_freq -1
```

For Humanoid environment:
```
python rob831/scripts/run_hw1.py --expert_policy_file rob831/policies/experts/Humanoid.pkl --env_name Humanoid-v2 --exp_name bc_humanoid --n_iter 1 --expert_data rob831/expert_data/expert_data_Humanoid-v2.pkl --video_log_freq -1
```

For Walker environment:
```
python rob831/scripts/run_hw1.py --expert_policy_file rob831/policies/experts/Walker2d.pkl --env_name Walker2d-v2 --exp_name bc_walker --n_iter 1 --expert_data rob831/expert_data/expert_data_Walker2d-v2.pkl --video_log_freq -1
```

For Hopper environment:

```
python rob831/scripts/run_hw1.py --expert_policy_file rob831/policies/experts/Hopper.pkl --env_name Hopper-v2 --exp_name bc_hopper --n_iter 1 --expert_data rob831/expert_data/expert_data_Hopper-v2.pkl --video_log_freq -1
```

For Half-Cheetah environment:

```
python rob831/scripts/run_hw1.py --expert_policy_file rob831/policies/experts/HalfCheetah.pkl --env_name HalfCheetah-v2 --exp_name bc_cheetah --n_iter 1 --expert_data rob831/expert_data/expert_data_HalfCheetah-v2.pkl --video_log_freq -1
```

Question1.3: 

For Ant environment:
```
python rob831/scripts/run_hw1.py --expert_policy_file rob831/policies/experts/Ant.pkl --env_name Ant-v2 --exp_name bc_ant --n_iter 1 --expert_data rob831/expert_data/expert_data_Ant-v2.pkl --video_log_freq -1 --ep_len 1000 --eval_batch_size 5000
```

For Humanoid environment:
```
python rob831/scripts/run_hw1.py --expert_policy_file rob831/policies/experts/Humanoid.pkl --env_name Humanoid-v2 --exp_name bc_humanoid --n_iter 1 --expert_data rob831/expert_data/expert_data_Humanoid-v2.pkl --video_log_freq -1 --ep_len 1000 --eval_batch_size 5000

```


Question1.4: 

```
python rob831/scripts/hyperparameter_steps.py --expert_policy_file rob831/policies/experts/Ant.pkl --env_name Ant-v2 --exp_name bc_ant --n_iter 1 --expert_data rob831/expert_data/expert_data_Ant-v2.pkl --video_log_freq -1
```

## DAgger

For Ant environment:

```
python rob831/scripts/dagger.py --expert_policy_file rob831/policies/experts/Ant.pkl --env_name Ant-v2 --exp_name dagger_ant --n_iter 10 --do_dagger --expert_data rob831/expert_data/expert_data_Ant-v2.pkl --video_log_freq -1
```

For Humanoid environment:
```
python rob831/scripts/dagger.py --expert_policy_file rob831/policies/experts/Humanoid.pkl --env_name Humanoid-v2 --exp_name dagger_humanoid --n_iter 10 --do_dagger --expert_data rob831/expert_data/expert_data_Humanoid-v2.pkl --video_log_freq -1
```

