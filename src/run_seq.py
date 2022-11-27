import os
import sys
import argparse

from src.lib.io import writeJSON, readJSON
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
    parser.add_argument("-r", "--remove", type=str)
    args = parser.parse_args()

    if not os.path.exists(PORTS_FILE):
        writeJSON(PORTS_FILE, {})

    ports = readJSON(PORTS_FILE)

    if args.info:
        printTable([{"port": key, "program": ports[key]} for key in ports], ["port", "program"])
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


if __name__ == "__main__":
    main()
