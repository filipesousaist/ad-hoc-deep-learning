import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("agent", choices=["npc", "drqn"])
    parser.add_argument("-d", "--display", action="store_true", help="launch HFO visualizer")
    parser.add_argument("-g", "--gnome-terminal", action="store_true", help="lauch agent in an external terminal")
    parser.add_argument("-l", "--load-path", type=str)
    parser.add_argument("-a", "--agent-state-path", type=str)
    parser.add_argument("-t", "--test-output-path", type=str)
    parser.add_argument("-T", "--train-output-path", type=str)
    args = parser.parse_args()
    
    players_arg = "--offense-agents" if args.agent == "drqn" else "--offense-npcs" 
    display_arg = "--no-sync" if args.display else "--headless"
    os.system("LC_ALL=C ../HFO/bin/HFO --frames-per-trial 200 --untouched-time 200 {} 1 {} &".format(players_arg, display_arg))

    if args.agent == "drqn":
        gnome_terminal_command = "gnome-terminal -x " if args.gnome_terminal else ""

        load_file_arg = " --load " + args.load_path if args.load_path else ""
        agent_state_path_arg = " --agent-state-path " + args.agent_state_path if args.agent_state_path else ""
        test_output_path_arg = " --test-output-path " + args.test_output_path if args.test_output_path else ""
        train_output_path_arg = " --train-output-path " + args.train_output_path if args.train_output_path else ""

        os.system("{}script -c 'python drqn-offense-agent-for-1v0.py{}{}{}{}' output/output.txt".format(
            gnome_terminal_command, load_file_arg, agent_state_path_arg, test_output_path_arg, train_output_path_arg))
    
    while not input().lower().startswith('q'):
        pass

    os.system("killall rcssserver -9")
    os.system("killall python")
    
if __name__ == '__main__':
    main()
