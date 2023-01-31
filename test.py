import os

import numpy as np
from models import DQN, AtariViT, DNN
import retro
from retro_wrappers import make_atari_env
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.policy import DQNPolicy, RandomPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.env import ShmemVectorEnv
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.discrete import IntrinsicCuriosityModule

import re 
import time
import gym 
import logging
from collections import defaultdict
import pickle
import fire
from memory_profiler import profile

logging.basicConfig(level = logging.INFO)



GENRES = \
'''
- Action
- Adventure
- Puzzle
- Sports
- Shooting
- Fighting
- Platformer
- Racing
- Educational
- Simulation
- Arcade 
- Strategy
- RPG
'''.strip().replace('\n','').split('-')[1:]

GENRES = [genre.strip().lower() for genre in GENRES]

LIST_OF_GAMES = \
    """
    Importing Adventure-Atari2600
    Importing AirRaid-Atari2600
    Importing Alien-Atari2600
    Importing Amidar-Atari2600
    Importing Asteroids-Atari2600
    Importing BattleZone-Atari2600
    Importing BeamRider-Atari2600
    Importing Berzerk-Atari2600
    Importing Bowling-Atari2600
    Importing Boxing-Atari2600
    Importing Breakout-Atari2600
    Importing Carnival-Atari2600
    Importing Centipede-Atari2600
    Importing ChopperCommand-Atari2600
    Importing CrazyClimber-Atari2600
    Importing ElevatorAction-Atari2600
    Importing Enduro-Atari2600
    Importing FishingDerby-Atari2600
    Importing Freeway-Atari2600
    Importing Frostbite-Atari2600
    Importing Hero-Atari2600
    Importing Jamesbond-Atari2600
    Importing JourneyEscape-Atari2600
    Importing Kangaroo-Atari2600
    Importing MontezumaRevenge-Atari2600
    Importing Phoenix-Atari2600
    Importing PrivateEye-Atari2600
    Importing Qbert-Atari2600
    Importing Robotank-Atari2600
    Importing Seaquest-Atari2600
    Importing Assault-Atari2600
    Importing Solaris-Atari2600
    Importing StarGunner-Atari2600
    Importing TimePilot-Atari2600
    Importing UpNDown-Atari2600
    Importing Pong-Atari2600
    Importing Zaxxon-Atari2600
    """

LIST_OF_GAMES_SNES = """
    Importing ZoolNinjaOfTheNthDimension-Snes
    Importing AcceleBrid-Snes
    Importing ActionPachio-Snes
    Importing ActRaiser2-Snes
    Importing AddamsFamilyPugsleysScavengerHunt-Snes
    Importing AddamsFamily-Snes
    Importing AdventuresOfDrFranken-Snes
    Importing AdventuresOfKidKleets-Snes
    Importing AdventuresOfMightyMax-Snes
    Importing AdventuresOfRockyAndBullwinkleAndFriends-Snes
    Importing AdventuresOfYogiBear-Snes
    Importing AeroFighters-Snes
    Importing AeroTheAcroBat-Snes
    Importing AeroTheAcroBat2-Snes
    Importing AirCavalry-Snes
    Importing AlfredChicken-Snes
    Importing AlienVsPredator-Snes
    Importing ArcherMacleansSuperDropzone-Snes
    Importing ArdyLightfoot-Snes
    Importing ArtOfFighting-Snes
    Importing Asterix-Snes
    Importing Axelay-Snes
    Importing BOB-Snes
    Importing BatmanReturns-Snes
"""
def main(rerun : bool = False): 
    

    all_games = re.findall(r"Importing (.*-Snes)",LIST_OF_GAMES_SNES)
    sample_dict = get_sample_freq(all_games,True)

    logging.info(f"Retro render sampling dict: {sample_dict}")
    #exit(0)
    #get_tianshou_stats(all_games,1)
    #exit(0)
    """
    if not rerun and os.path.exists('./genre_dict.pkl'): 
        try: 
            with open('genre_dict.pkl', 'rb') as f:
                genre_dict = pickle.load(f)
        except: 
            genre_dict = {}
            pass
    else: 
        genre_dict = get_game_genres(all_games)

    genre_count = {genre:len(games) for genre,games in genre_dict.items()}

    logging.info(f"Genre Breakdown: {genre_count}")
    """


    for render in [True,False]: 
        post_fix = 'render' if render else 'no_render'
        if not rerun and os.path.exists(f'./sample_dict_{post_fix}.pkl'): 
            try: 
                with open(f'./sample_dict_{post_fix}.pkl', 'rb') as f:
                    sample_dict = pickle.load(f)
            except: 
                sample_dict = {}
                pass
        else: 
            sample_dict = get_sample_freq(all_games,render)

        logging.info(f"Retro {post_fix} sampling dict: {sample_dict}")

        
    for render in [False,True]: 
        post_fix = 'render' if render else 'no_render'
        if not rerun and os.path.exists(f'./tsample_dict_{post_fix}.pkl'): 
            try: 
                with open(f'./tsample_dict_{post_fix}.pkl', 'rb') as f:
                    tsample_dict = pickle.load(f)
            except: 
                tsample_dict = {}
                pass
        else: 
            train_num_dict = defaultdict(dict)
            for training_num in [1,2,4,8,16,32,64]: 
                temp_dict = get_tianshou_stats(all_games, training_num, render)

                for game,t in temp_dict.items(): train_num_dict[game][training_num]=t

            with open(f'./tsample_dict_{post_fix}.pkl', 'wb') as f:
                pickle.dump(train_num_dict, f)
            logging.info(f"Tianshou {post_fix} sampling dict: {train_num_dict}")
            

    
        

def get_game_genres(all_games,emu='Atari2600'): 

    genre_game_dict = defaultdict(list) 
    for game in all_games: 
        logging.info("Running game: {} for classification".format(game))
        env = retro.make(game=game)
        env.reset()
        for _ in range(500): 
            obs, rew, done, info = env.step(env.action_space.sample())
            # rew will be a list of [player_1_rew, player_2_rew]
            # done and info will remain the same
            #if done:
            #    obs = env.reset()
            env.render() 
    # CTRL + C to move onto next game 
        while True: 
            genre = input("Input the game genre of {}:\n".format(game))
            if not all([split in GENRES for split in genre.strip().lower().split(',')]): 
                logging.warning("Please choose genre(s) from the following list", GENRES)
            else: 
                break
        env.close()
        for split in genre.strip().lower().split(','): 
            genre_game_dict[split].append(game) 
    
    with open('genre_dict_{}.pkl'.format(emu), 'wb') as f:
        pickle.dump(genre_game_dict, f)

    return genre_game_dict

def get_sample_freq(all_games, render = False, emu='Atari2600'): 
    game_freq_dict = defaultdict()
    render=False

    if render: 
        post_fix = 'render'
    else: 
        post_fix = 'no_render'

    for game in all_games: 
        logging.info(f"Running game: {game} for measuring sampling rate")
        env = retro.make(game=game) 
        game_freq_dict[game] = 0
        try: 
            for i in range(1): 
                start = time.time()
                env.reset() 
                for _ in range(500):
                    obs, rew, done, info = env.step(env.action_space.sample())
                    # rew will be a list of [player_1_rew, player_2_rew]
                    # done and info will remain the same
                    #if done:
                    if render: 
                        env.render() 
                end = time.time()
                game_freq_dict[game] += float(end-start)
            env.close()
            # simple averaging to get frequency 
            game_freq_dict[game] = (500 * 1) / game_freq_dict[game]
            logging.info(f"Average sampling rate in {post_fix} mode: {game_freq_dict[game]}")

        except: 
            raise Exception(f"Not sure what happened. Died in execution of {game} in sampling")



    with open('sample_dict_{}_{}.pkl'.format(post_fix,emu), 'wb') as f:
        pickle.dump(game_freq_dict, f)

    return game_freq_dict

#@profile
def get_tianshou_stats(all_games, training_num, render= True): 

    game_tsample_freq = defaultdict()
    for game in ['Pong-Atari2600']: 
        train_env = ShmemVectorEnv(
            [
                lambda:
                retro.make(game)
                for _ in range(training_num)
            ]
        )

        train_collector = Collector(RandomPolicy(), train_env)

        # Only need to have the collector collect x samples for profiling, no need to iterate
        
        start = time.time()
        train_collector.collect(n_step=5000, random= True, render=0. if render else None) 
        end = time.time() 

        game_tsample_freq[game] = float(end-start)

    return game_tsample_freq







            

    env = retro.make(game="Zaxxon-Atari2600")
    #env = gym.make("Pong")
    print(env.metadata)
    obs = env.reset()
    while True: 
        start = time.time()
        for _ in range(10000):
            # action_space will by MultiBinary(16) now instead of MultiBinary(8)
            # the bottom half of the actions will be for player 1 and the top half for player 2
            obs, rew, done, info = env.step(env.action_space.sample())
            # rew will be a list of [player_1_rew, player_2_rew]
            # done and info will remain the same
            if done:
                obs = env.reset()
            #env.render()
        end = time.time()
        print(end-start)
    env.close()

if __name__ == '__main__': 
    

    #for training_num in [1,2,4,8,16,32,64]: 
    #    temp_dict = get_tianshou_stats(None,training_num=training_num, render=False)
    #for training_num in [1,2,4,8,16,32]: 
        #temp_dict = get_tianshou_stats(None,training_num=training_num, render=True)
    yes = retro.make('Asteroids-Atari2600')
    print(yes.action_space,file=open('lul.txt','w'))
    space = yes.action_space
    #print(yes.get_action_meaning(space.sample()))
    print("VALID_ACTIONS",yes.button_combos)
    for a in yes.data.valid_actions(): 
        print(yes.get_action_meaning(a))
    print(np.array(yes.get_screen()).shape)
    print(yes.buttons)
    print(yes.unwrapped.buttons)
    yes.reset()

    while False: 
        y = yes.action_space.sample()
        yes.step(y)
        yes.render() 
        print("DID ACTION ",yes.get_action_meaning(y))
    yes.close()
    #fire.Fire(main)