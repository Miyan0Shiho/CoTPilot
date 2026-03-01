import os
import sys

# Add local libraries to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(os.path.join(project_root, "opencompass"))
sys.path.append(os.path.join(project_root, "EvoPrompt"))
sys.path.append(current_dir)

from cot_pilot.interface.cli import CLI

def main():
    cli = CLI()
    cli.start()

if __name__ == "__main__":
    main()
