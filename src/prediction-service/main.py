from aare.utils import find_project_root
from lib import hello

if __name__ == "__main__":
    hello()
    print(f"Project Root: {find_project_root()}")
