import os
import sys
import re
import argparse

from src.lib.io import writeJSON, readJSON, readTxt
from src.lib.paths import DEFAULT_PORT, DEFAULT_DIRECTORY, getPath
from src.lib.io import printTable

PORTS_FILE = "run_seq_ports.json"
DEFAULT_SAVE_PERIOD = 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-D", "--directory", type=str)
    parser.add_argument("-p", "--port", type=str)
    parser.add_argument("-s", "--save-period", type=int)
    parser.add_argument("-i", "--info", action="store_true")
    parser.add_argument("-e", "--experiments", type=str)
    parser.add_argument("-S", "--status", type=str)
    parser.add_argument("-R", "--regular-expression", type=str)
    parser.add_argument("-r", "--remove", type=str)
    args = parser.parse_args()

    if not os.path.exists(PORTS_FILE):
        writeJSON(PORTS_FILE, {})

    ports = readJSON(PORTS_FILE)

    if sum([bool(flag) for flag in (args.info, args.experiments, args.remove)]) > 1:
        exit("At most one of --info, --remove and --experiments can be set.")

    if args.info:
        printTable([{"port": key, "program": ports[key]} for key in ports], ["port", "program"])
    elif args.experiments:
        experiments_list = []
        getExperiments(args.experiments, experiments_list)
        if args.status:
            experiments_list = list(filter(lambda item: item["status"] == args.status, experiments_list))
        if args.regular_expression:
            pattern = re.compile(args.regular_expression)
            experiments_list = list(filter(lambda item: re.match(pattern, item["path"]), experiments_list))
        experiments_list = sorted(experiments_list, key=lambda item: item["path"])
        printTable(experiments_list, ["path", "status"])
    elif args.remove:
        removePort(ports, args.remove)
    else:
        directory = args.directory or DEFAULT_DIRECTORY
        port = args.port or DEFAULT_PORT
        save_period = args.save_period or DEFAULT_SAVE_PERIOD

        if port in ports:
            exit(f"Port {port} already in use by {ports[port]}")

        if os.path.exists(getPath(directory, "save")):
            exit(f"Found save data in directory '{directory}'")

        ports[port] = " ".join(sys.argv)
        writeJSON(PORTS_FILE, ports)

        os.system(f"python -m src.seq_evaluate -D {directory} -s {save_period} -p {port}")

        ports = readJSON(PORTS_FILE)
        removePort(ports, port)


def removePort(ports, port):
    del ports[port]
    writeJSON(PORTS_FILE, ports)

    print(f"Successfully removed port {port}")


def getExperiments(directory, experiments_list):
    files = os.listdir(directory)
    if os.path.exists(getPath(directory, "input")):
        status = "New"
        save_path = getPath(directory, "save")
        if os.path.exists(save_path):
            if readTxt(save_path)["next_episode"] == "44050":
                status = "Finished"
            else:
                status = "In Progress"
        experiments_list.append({
            "path": directory,
            "status": status
        })
    for file in files:
        path = os.path.join(directory, file)
        if os.path.isdir(path):
            getExperiments(path, experiments_list)



if __name__ == "__main__":
    main()
