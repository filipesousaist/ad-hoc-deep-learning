DEFAULT_PORT = 6000
DEFAULT_DIRECTORY = "./output"
INPUT_FILE_NAME = "input.json"
OUTPUT_FILE_NAME = "output.txt"
TEST_OUTPUT_FILE_NAME = "TEST-results.txt"
TRAIN_OUTPUT_FILE_NAME = "TRAIN-results.txt"
AGENT_STATE_FILE_NAME = "agent-state"
SAVE_FILE_NAME = "save.txt"

_paths = {
    "input":        INPUT_FILE_NAME,
    "output":       OUTPUT_FILE_NAME,
    "test-output":  TEST_OUTPUT_FILE_NAME,
    "train-output": TRAIN_OUTPUT_FILE_NAME,
    "agent-state":  AGENT_STATE_FILE_NAME,
    "save":         SAVE_FILE_NAME
}


def getPath(directory, type: str) -> dict:
    return directory.rstrip("/") + "/" + _paths[type]