#!/bin/bash

python3 retro_main.py --policy_network DQN 
python3 retro_main.py --policy_network DNN 
# ..... PC can't handle 20 envs when running ViT :(
python3 retro_main.py --policy_network ViT --training-num 1 --test-num 1