import multiprocessing


class Result:
    def __init__(self, value: int, result: "Result" = None):
        self.value = value
        self.result = result

    def __str__(self):
        return f"{{Value: {self.value}, Result: {self.result}}}"


class MyProcess(multiprocessing.Process):
    def __init__(self, result_queue: multiprocessing.JoinableQueue, first_result: Result):
        super().__init__()
        self.result_queue = result_queue
        self.first_result = first_result
        self.other_result = Result(350, Result(75))


    def run(self) -> None:
        self.result_queue.put(self.first_result)
        self.result_queue.put(Result(24, Result(38)))

    def getResult(self) -> Result:
        return self.other_result


q = multiprocessing.JoinableQueue()
p = MyProcess(q, Result(-37))
p.start()

print(p.getResult())

q.join()

for i in range(2):
    print(q.get())

p.terminate()

print(p.getResult())
