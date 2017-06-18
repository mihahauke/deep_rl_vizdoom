import argparse

# TODO why isn't it in constants?
DEFAULT_SETTINGS_FILE = "settings/basic.yml"

# Help messages:
SETTINGS_HELP_MSG = "load settings from yaml files. " \
                    "If multiple files are specified, overlapping settings " \
                    "will be overwritten according to order of appearance " \
                    "(e.g. settings from file #1 will be overwritten by file #2)."
Q_HELP_MSG = "use n-step qlearning instead of a3c"


def _add_commons(parser):
    parser.add_argument("--settings", "-s",
                        dest="settings_yml",
                        metavar='YAML_FILE',
                        nargs="*",
                        type=str,
                        default=[DEFAULT_SETTINGS_FILE],
                        help=SETTINGS_HELP_MSG)
    # TODO run
    # TODO tag
    # TODO tags extension


def parse_train_a3c_args():
    parser = argparse.ArgumentParser(description='A3C implementation for ViZDoom in Tensorflow.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _add_commons(parser)

    return parser.parse_args()


def parse_train_adqn_args():
    parser = argparse.ArgumentParser(description='Asynchronous n-step DQN implementation for ViZDoom in Tensorflow.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _add_commons(parser)

    return parser.parse_args()


def parse_train_dqn_args():
    parser = argparse.ArgumentParser(description='DQN implementation for ViZDoom in Tensorflow.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _add_commons(parser)

    return parser.parse_args()
