import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from fs2.cli import train_main
if __name__ == "__main__": train_main(pathlib.Path(__file__).resolve().parent)
