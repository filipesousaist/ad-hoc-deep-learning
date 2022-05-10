import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("agent", choices=["npc", "drqn"])
    parser.add_argument("-d", "--display", action="store_true")
    parser.add_argument("-t", "--terminal", action="store_true")
    parser.add_argument("-l", "--load", type=str)
    args = parser.parse_args()
    
    playersArg = "--offense-agents" if args.agent == "drqn" else "--offense-npcs" 
    displayArg = "--no-sync" if args.display else "--headless"
    os.system("LC_ALL=C ../HFO/bin/HFO {} 1 {} &".format(playersArg, displayArg))

    if args.agent == "drqn":
        loadArg = " --load " + args.load if args.load else ""
        terminalCommand = "gnome-terminal -x " if args.terminal else ""
        os.system("{}script -c 'python drqn-offense-agent-for-1v0.py{}' output/output.txt".format(terminalCommand, loadArg))
    
    while not input().lower().startswith('q'):
        pass

    os.system("killall rcssserver -9")
    os.system("killall python")
    
if __name__ == '__main__':
    main()
