import os
import argparse

DEFAULT_OUTPUT_PATH = "./output"
OUTPUT_FILE_NAME = "output.txt"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("agent", choices=["npc", "drqn"], help = "the type of agent to evaluate")
    parser.add_argument("-v", "--visualizer", action="store_true", help="launch HFO visualizer")
    parser.add_argument("-g", "--gnome-terminal", action="store_true", help="lauch agent in an external terminal")
    
    parser.add_argument("-l", "--load", action="store_true")
    parser.add_argument("-e", "--train-episode", type=int)
    parser.add_argument("-t", "--test-only", action="store_true")
    parser.add_argument("-f", "--custom-features", action="store_true")
    parser.add_argument("-O", "--output-path", type=str)

    parser.add_argument("-o", "--offense-players", type=int,
        help="number of players in the offense team, including the agent to be evaluated")
    parser.add_argument("-d", "--defense-players", type=int,
        help="number of players in the defense team")

    args = parser.parse_args()

    num_offense_players = args.offense_players or 1
    if num_offense_players < 1:
        print("At least 1 player is needed")

    num_offense_agents = 0 if args.agent == "npc" else 1
    num_offense_npcs = num_offense_players - num_offense_agents
    num_defense_npcs = args.defense_players or 0 # All defense players are npcs
    
    visualizer_arg = "--no-sync" if args.visualizer else "--headless"
    os.system("LC_ALL=C ../HFO/bin/HFO --frames-per-trial 500 --untouched-time 200 {} --offense-agents {} --offense-npcs {} --defense-npcs {} &".format(
        visualizer_arg, num_offense_agents, num_offense_npcs, num_defense_npcs))

    if args.agent == "drqn":
        gnome_terminal_command = "gnome-terminal -x " if args.gnome_terminal else ""

        load_arg = " --load" if args.load else ""
        train_episode_arg = " --train-episode {}".format(args.train_episode) if args.train_episode else ""
        test_only_arg = " --test-only" if args.test_only else ""
        custom_features_arg = " --custom-features" if args.custom_features else ""
        output_path = (args.output_path or DEFAULT_OUTPUT_PATH).rstrip("/")
        output_path_arg = " --output-path " + output_path

        os.system("{}script -c 'python drqn-offense-agent-for-1v0.py{}{}{}{}{}' {}/{}".format(
            gnome_terminal_command, load_arg, train_episode_arg,
            test_only_arg, custom_features_arg, output_path_arg,
            output_path, OUTPUT_FILE_NAME))
    
    while not input().lower().startswith('q'):
        pass

    os.system("killall rcssserver -9")
    os.system("killall python")
    
if __name__ == '__main__':
    main()
