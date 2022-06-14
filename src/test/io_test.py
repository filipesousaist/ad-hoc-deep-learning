from src.lib.io import logOutput
from sys import argv

logOutput("./src/test/log.txt")

print(argv[1])
print("halfways")
print(1/0)