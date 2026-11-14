import copy
import random
import numpy as np
import yaml
import os
import argparse
import json
from pathlib import Path
from rich import print

import gymnasium as gym
import highway_env  # register highway-v0 explicitly
from gymnasium.wrappers import RecordVideo

from dilu.scenario.envScenario import EnvScenario
from dilu.driver_agent.driverAgent import DriverAgent
from dilu.driver_agent.reflectionAgent import ReflectionAgent
from dilu.api_config import configure_api


test_list_seed = [5838, 2421, 7294, 9650, 4176, 6382, 8765, 1348,
                  4213, 2572, 5678, 8587, 512, 7523, 6321, 5214, 31]


def setup_env(config, offline=False):
    if not offline:
        configure_api(config)

    # environment setting
    env_config = {
        'highway-v0':
        {
            "observation": {
                "type": "Kinematics",
                "features": ["presence", "x", "y", "vx", "vy"],
                "absolute": True,
                "normalize": False,
                "vehicles_count": config["vehicle_count"],
                "see_behind": True,
            },
            "action": {
                "type": "DiscreteMetaAction",
                "target_speeds": np.linspace(5, 32, 9),
            },
            "lanes_count": 4,
            "other_vehicles_type": config["other_vehicle_type"],
            "duration": config["simulation_duration"],
            "vehicles_density": config["vehicles_density"],
            "show_trajectories": True,
            "render_agent": True,
            "offscreen_rendering": True,
            "scaling": 5,
            'initial_lane_id': None,
            "ego_spacing": 4,
        }
    }

    return env_config


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description='DiLu small-scale reproduction runner')
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.yaml'))
    parser.add_argument('--output', type=Path, required=True,
                        help='New/empty directory under the parent experiment/ directory')
    parser.add_argument('--offline-smoke', action='store_true',
                        help='One simulation step with IDLE; no LLM, memory or API calls')
    args = parser.parse_args()
    with args.config.open(encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    output = args.output.resolve()
    experiment_root = Path(__file__).resolve().parents[2] / 'experiment'
    if experiment_root not in output.parents:
        parser.error('--output must be inside ' + str(experiment_root))
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error('--output must be a new or empty directory; existing results are preserved')
    if any(not isinstance(config.get(k), int) or isinstance(config[k], bool) or config[k] < 1
           for k in ('episodes_num', 'simulation_duration')):
        parser.error('episodes_num and simulation_duration must be positive integers')
    if not isinstance(config.get('few_shot_num'), int) or config['few_shot_num'] < 0:
        parser.error('few_shot_num must be a nonnegative integer')
    seeds = config.get('seeds')
    if seeds is not None and (not isinstance(seeds, list) or not seeds or
                              any(type(seed) is not int for seed in seeds)):
        parser.error('seeds must be a nonempty list of integers')
    if args.offline_smoke:
        config.update(episodes_num=1, simulation_duration=1, few_shot_num=0,
                      reflection_module=False)
    env_config = setup_env(config, offline=args.offline_smoke)

    REFLECTION = config["reflection_module"]
    memory_path = str((Path(__file__).parent / config["memory_path"]).resolve())
    few_shot_num = config["few_shot_num"]
    result_folder = str(output)
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    with open(result_folder + "/" + 'log.txt', 'w') as f:
        f.write("memory_path {} | result_folder {} | few_shot_num: {} | lanes_count: {} \n".format(
            memory_path, result_folder, few_shot_num, env_config['highway-v0']['lanes_count']))

    agent_memory = None
    if few_shot_num > 0 or REFLECTION:
        if os.getenv('DILU_CHAT_PROVIDER') == 'deepseek' and not os.getenv('DILU_EMBEDDING_API_KEY'):
            raise ValueError('Set a separate DILU_EMBEDDING_API_KEY for memory retrieval/reflection.')
        from dilu.driver_agent.vectorStore import DrivingMemory
        if few_shot_num > 0 and not Path(memory_path).is_dir():
            raise ValueError('Memory directory does not exist: ' + memory_path)
        agent_memory = DrivingMemory(db_path=memory_path)
    if REFLECTION:
        updated_memory = DrivingMemory(db_path=str(output / 'memory_updated'))
        updated_memory.combineMemory(agent_memory)

    episode = 0
    while episode < config["episodes_num"]:
        # setup highway-env
        envType = 'highway-v0'
        env = gym.make(envType, render_mode="rgb_array")
        env.configure(env_config[envType])
        result_prefix = f"highway_{episode}"
        env = RecordVideo(env, result_folder, name_prefix=result_prefix)
        env.unwrapped.set_record_video_wrapper(env)
        seed = seeds[episode % len(seeds)] if seeds else random.choice(test_list_seed)
        obs, info = env.reset(seed=seed)
        env.render()

        # scenario and driver agent setting
        database_path = result_folder + "/" + result_prefix + ".db"
        sce = EnvScenario(env, envType, seed, database_path)
        DA = None if args.offline_smoke else DriverAgent(sce, verbose=True)
        if REFLECTION:
            RA = ReflectionAgent(verbose=True)

        response = "Not available"
        action = "Not available"
        docs = []
        collision_frame = -1

        try:
            already_decision_steps = 0
            for i in range(0, config["simulation_duration"]):
                obs = np.array(obs, dtype=float)

                print("[cyan]Retreive similar memories...[/cyan]")
                fewshot_results = agent_memory.retriveMemory(
                    sce, i, few_shot_num) if few_shot_num > 0 else []
                fewshot_messages = []
                fewshot_answers = []
                fewshot_actions = []
                for fewshot_result in fewshot_results:
                    fewshot_messages.append(
                        fewshot_result["human_question"])
                    fewshot_answers.append(fewshot_result["LLM_response"])
                    fewshot_actions.append(fewshot_result["action"])
                    mode_action = max(
                        set(fewshot_actions), key=fewshot_actions.count)
                    mode_action_count = fewshot_actions.count(mode_action)
                if few_shot_num == 0:
                    print("[yellow]Now in the zero-shot mode, no few-shot memories.[/yellow]")
                else:
                    print("[green4]Successfully find[/green4]", len(
                        fewshot_actions), "[green4]similar memories![/green4]")

                sce_descrip = sce.describe(i)
                avail_action = sce.availableActionsDescription()
                print('[cyan]Scenario description: [/cyan]\n', sce_descrip)
                # print('[cyan]Available actions: [/cyan]\n',avail_action)
                if args.offline_smoke:
                    action, response, human_question, fewshot_answer = 1, 'offline IDLE smoke', sce_descrip, ''
                else:
                    action, response, human_question, fewshot_answer = DA.few_shot_decision(
                        scenario_description=sce_descrip, available_actions=avail_action,
                        previous_decisions=action,
                        fewshot_messages=fewshot_messages,
                        driving_intensions="Drive safely and avoid collisons",
                        fewshot_answers=fewshot_answers,
                    )
                docs.append({
                    "sce_descrip": sce_descrip,
                    "human_question": human_question,
                    "response": response,
                    "action": action,
                    "sce": copy.deepcopy(sce)
                })

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                already_decision_steps += 1

                env.render()
                sce.promptsCommit(i, None, done, human_question,
                                  fewshot_answer, response)

                print("--------------------")

                if done:
                    print("[red]Simulation ended after running steps: [/red] ", i)
                    if env.unwrapped.vehicle.crashed:
                        collision_frame = i
                    break
        finally:

            with open(result_folder + "/" + 'log.txt', 'a') as f:
                f.write(
                    "Simulation {} | Seed {} | Steps: {} | File prefix: {} \n".format(episode, seed, already_decision_steps, result_prefix))
                
            if REFLECTION:
                print("[yellow]Now running reflection agent...[/yellow]")
                if collision_frame != -1: # End with collision
                    for i in range(collision_frame, -1, -1):
                        if docs[i]["action"] != 4:  # not decelearate
                            corrected_response = RA.reflection(
                                docs[i]["human_question"], docs[i]["response"])
                            
                            choice = input("[yellow]Do you want to add this new memory item to update memory module? (Y/N): ").strip().upper()
                            if choice == 'Y':
                                updated_memory.addMemory(
                                    docs[i]["sce_descrip"],
                                    docs[i]["human_question"],
                                    corrected_response,
                                    docs[i]["action"],
                                    docs[i]["sce"],
                                    comments="mistake-correction"
                                )
                                print("[green] Successfully add a new memory item to update memory module.[/green]. Now the database has ", len(
                                    updated_memory.scenario_memory._collection.get(include=['embeddings'])['embeddings']), " items.")
                            else:
                                print("[blue]Ignore this new memory item[/blue]")
                            break
                else:
                    print("[yellow]Do you want to add[/yellow]",len(docs)//5, "[yellow]new memory item to update memory module?[/yellow]",end="")
                    choice = input("(Y/N): ").strip().upper()
                    if choice == 'Y':
                        cnt = 0
                        for i in range(0, len(docs)):
                            if i % 5 == 1:
                                updated_memory.addMemory(
                                    docs[i]["sce_descrip"],
                                    docs[i]["human_question"],
                                    docs[i]["response"],
                                    docs[i]["action"],
                                    docs[i]["sce"],
                                    comments="no-mistake-direct"
                                )
                                cnt +=1
                        print("[green] Successfully add[/green] ",cnt," [green]new memory item to update memory module.[/green]. Now the database has ", len(
                                    updated_memory.scenario_memory._collection.get(include=['embeddings'])['embeddings']), " items.")
                    else:
                        print("[blue]Ignore these new memory items[/blue]")
            

            print("==========Simulation {} Done==========".format(episode))
            episode += 1
            env.close()
    with (output / 'run_info.json').open('w', encoding='utf-8') as stream:
        json.dump({'mode': 'offline-smoke' if args.offline_smoke else 'api',
                   'provider': config.get('OPENAI_API_TYPE', 'deepseek'),
                   'model': os.getenv('OPENAI_CHAT_MODEL') if not args.offline_smoke else None,
                   'episodes': episode, 'seeds': seeds,
                   'few_shot_num': few_shot_num, 'reflection': REFLECTION}, stream, indent=2)
