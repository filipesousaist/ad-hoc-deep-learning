import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("agent", choices=["npc", "drqn"])
    parser.add_argument("-d", "--display", action="store_true")
    args = parser.parse_args()
    
    playersArg = "--offense-agents" if args.agent == "drqn" else "--offense-npcs" 
    displayArg = "--no-sync" if args.display else "--headless"
    os.system("LC_ALL=C ../HFO/bin/HFO {} 1 {} &".format(playersArg, displayArg))

    if args.agent == "drqn":
        os.system("gnome-terminal -x script -c 'python drqn-offense-agent-for-1v0.py' output/output.txt")
    
    while not input().lower().startswith('q'):
        pass

    os.system("killall rcssserver -9")
    os.system("killall python")
    
if __name__ == '__main__':
    main()
