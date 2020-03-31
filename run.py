import argparse
import sys
import importlib


class ProcessRunner:
    """ This Class will run any process from the processes directory
        Use it using the following command:

        python runner.py --config path/to/config/yaml --process MyProcess start_date end_date
    """
    def __init__(self, cmdl_args):
        parser = argparse.ArgumentParser()
        parser.add_argument('level')
        parser.add_argument('--save')
        args = parser.parse_known_args(cmdl_args)[0]

        self.level = args.level
        self.save = args.save

    def run(self):
        module = importlib.import_module("Animate")
        process_class = getattr(module, "Animate")
        process_class(self.level).run()


if __name__ == "__main__":
    ProcessRunner(sys.argv[1:]).run()
