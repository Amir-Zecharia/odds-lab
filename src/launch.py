"""Launch the dashboard with its verified offline snapshot."""
import subprocess
import sys
from src import config
from src.bundle import bootstrap


def main():
    bootstrap()
    return subprocess.call([sys.executable, '-m', 'streamlit', 'run',
                            str(config.PROJECT_ROOT / 'app' / 'Home.py'), *sys.argv[1:]],
                           cwd=config.PROJECT_ROOT)


if __name__ == '__main__':
    raise SystemExit(main())
