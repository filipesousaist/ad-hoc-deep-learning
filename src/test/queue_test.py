from multiprocessing import JoinableQueue, Process
import numpy as np

MAX = 5


class OpponentProcess(Process):
    def __init__(self, my_queue: JoinableQueue, opponent_queue: JoinableQueue):
        super().__init__()
        self._my_queue = my_queue
        self._opponent_queue = opponent_queue

    def run(self) -> None:
        _my_move = np.array([0, 0])
        while _my_move[0] < MAX:
            print("[Opponent] You moved ", self._opponent_queue.get())
            _my_move[0] += 1
            _my_move[1] += 3
            self._my_queue.put(_my_move)
            print("[Opponent] I moved", _my_move, flush=True)

        print("[Opponent] You moved ", self._opponent_queue.get())
        print("[Opponent] You moved ", self._opponent_queue.get())


def main():
    my_queue = JoinableQueue(1)
    opponent_queue = JoinableQueue(1)

    process = OpponentProcess(opponent_queue, my_queue)
    process.start()

    my_move = 0
    opponent_move = np.array([-1, -1])
    while opponent_move[0] < MAX:
        my_move_str = chr(ord('A') + my_move)
        my_queue.put(my_move_str)
        print("[Me] I moved", my_move_str)
        my_move += 1
        opponent_move = opponent_queue.get()
        print("[Me] You moved", opponent_move)

    my_queue.put("ONCE")
    print("[Me] I moved ONCE")
    my_queue.put("TWICE")
    print("[Me] I moved TWICE")


if __name__ == "__main__":
    main()
