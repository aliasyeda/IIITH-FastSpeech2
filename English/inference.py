import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from fs2.cli import inference_main
if __name__ == "__main__": inference_main(pathlib.Path(__file__).resolve().parent)
