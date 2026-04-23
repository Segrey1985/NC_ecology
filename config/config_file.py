import pathlib
from dataclasses import dataclass


@dataclass
class Config:
    BASE_DIR: str = pathlib.Path(__file__).resolve().parent.parent


cfg = Config()

if __name__ == "__main__":
    print(cfg)
