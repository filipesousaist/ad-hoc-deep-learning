DEFAULT_PORT = 6000
DEFAULT_DIRECTORY = "./output"
INPUT_FILE_NAME = "input.json"
OUTPUT_FILE_NAME = "output.txt"
TEST_OUTPUT_FILE_NAME = "TEST-results.txt"
TRAIN_OUTPUT_FILE_NAME = "TRAIN-results.txt"
AGENT_STATE_DIRECTORY_NAME = "agent-state"
SAVE_FILE_NAME = "save.txt"
LABEL_FILE_NAME = "label.txt"
KNOWLEDGE_DIRECTORY_NAME = "knowledge"
NN_MODEL_FILE_NAME = "nn-model.pkl"
PLASTIC_RESULTS = "PLASTIC-results.json"


_paths = {
    "input": INPUT_FILE_NAME,
    "output": OUTPUT_FILE_NAME,
    "test-output": TEST_OUTPUT_FILE_NAME,
    "train-output": TRAIN_OUTPUT_FILE_NAME,
    "agent-state": AGENT_STATE_DIRECTORY_NAME,
    "save": SAVE_FILE_NAME,
    "label": LABEL_FILE_NAME,
    "nn-model": NN_MODEL_FILE_NAME,
    "knowledge": KNOWLEDGE_DIRECTORY_NAME,
    "plastic-results": PLASTIC_RESULTS
}


def getPath(directory: str, purpose: str) -> str:
    return directory.rstrip("/") + ("/" if directory else "") + _paths[purpose]


def getAgentStatePath(directory: str, train_episode: int = -1, num_train_episodes: int = 1):
    return getPath(directory, "agent-state") + \
          ("/latest" if train_episode < 0 else
           f"/after{train_episode // num_train_episodes * num_train_episodes}episodes")
