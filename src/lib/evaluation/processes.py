import argparse
import os
import time
from signal import SIGTERM

from psutil import Popen, NoSuchProcess

from src.hfo_agents import is_custom, get_team_name
from src.lib.threads import WaitForQuitThread, TeammateThread, OpponentThread


def startProcesses(directory: str, port: int, args: argparse.Namespace, input_loadout: int, input_data: dict,
                   wait_for_quit_thread: WaitForQuitThread) -> Popen:
    hfo_process = launchHFO(input_data, port, args.gnome_terminal, args.visualizer)
    time.sleep(2)
    launchOtherAgents(directory, port, input_loadout, input_data, wait_for_quit_thread)
    return hfo_process


def launchHFO(input_data: dict, port: int, gnome_terminal: bool, visualizer: bool) -> Popen:
    gnome_terminal_command = "xterm -e " if gnome_terminal else ""

    background_process = "" if gnome_terminal else " &"

    agent_type = input_data["agent_type"]
    agent_is_binary = not is_custom(agent_type)

    teammates_type = input_data["teammates_type"]
    teammates_are_binary = not is_custom(teammates_type)

    opponents_type = input_data["opponents_type"]
    opponents_are_binary = not is_custom(opponents_type)

    offense_type = teammates_type if teammates_are_binary else agent_type
    offense_is_binary = agent_is_binary or teammates_are_binary

    if agent_is_binary and teammates_are_binary and agent_type != teammates_type:
        exit("[ERROR]: 'agent_type' and 'teammates_type' must be the same if both are binary.")

    num_offense_npcs = int(agent_is_binary) + input_data["num_teammates"] * int(teammates_are_binary)
    num_defense_npcs = input_data["num_opponents"] * int(opponents_are_binary)

    hfo_args = [
        " --no-logging",
        " --port {}".format(port) if port else "",
        " --no-sync" if visualizer else " --headless",
        " --fullstate" if input_data["fullstate"] else "",
        " --offense-agents {}".format(input_data["num_teammates"] + 1 - num_offense_npcs),
        " --offense-npcs {}".format(num_offense_npcs),
        " --offense-team {}".format(get_team_name(offense_type)) if offense_is_binary else "",
        " --defense-agents {}".format(input_data["num_opponents"] - num_defense_npcs),
        " --defense-npcs {}".format(num_defense_npcs),
        " --defense-team {}".format(get_team_name(opponents_type)) if opponents_are_binary else "",
        " --frames-per-trial {}".format(input_data["frames_per_trial"]),
        " --untouched-time {}".format(input_data["untouched_time"])
    ]

    unformatted_command = "{}../HFO/bin/HFO " + "{}" * len(hfo_args) + "{}"
    formatted_command = unformatted_command.format(gnome_terminal_command, *hfo_args, background_process)
    environment_variables = {**os.environ, "LC_ALL": "C"}

    return Popen(formatted_command, shell=True, env=environment_variables, start_new_session=True)


def launchOtherAgents(directory: str, port: int, input_loadout: int, input_data: dict,
                      wait_for_quit_thread: WaitForQuitThread) -> None:
    if is_custom(input_data["teammates_type"]):
        agent_type = input_data["agent_type"]
        team_name = "base" if is_custom(agent_type) else get_team_name(agent_type)
        for _ in range(int(input_data["num_teammates"])):
            TeammateThread(directory, port, input_loadout, input_data["teammates_type"],
                           team_name, wait_for_quit_thread).start()

    if is_custom(input_data["opponents_type"]):
        for _ in range(int(input_data["num_opponents"])):
            OpponentThread(directory, port, input_loadout, input_data["opponents_type"],
                           "base", wait_for_quit_thread).start()


def killProcesses(hfo_process: Popen, gnome_terminal: bool):
    if gnome_terminal:
        for p in hfo_process.children(recursive=True):
            try:
                p.kill()
            except NoSuchProcess:
                pass
        hfo_process.kill()
    else:
        os.killpg(os.getpgid(hfo_process.pid), SIGTERM)
        os.kill(hfo_process.pid, SIGTERM)
