import argparse
import sys
import importlib


class ProcessRunner:
    """ This Class will run any process from the processes directory
        Use it using the following command:

        python run.py --level 0 --save True --release True
    """
    def __init__(self, cmdl_args):
        parser = argparse.ArgumentParser()
        parser.add_argument('--level')
        parser.add_argument('--save')
        parser.add_argument('--release')
        args = parser.parse_known_args(cmdl_args)[0]

        self.level = args.level
        self.save = args.save
        self.release = args.release

    def run(self):
        module = importlib.import_module("Animate")
        process_class = getattr(module, "Animate")
        process_class(self.level, self.save, self.release).run()


if __name__ == "__main__":
    ProcessRunner(sys.argv[1:]).run()
