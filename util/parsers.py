import argparse

# TODO why isn't it in constants?
DEFAULT_SETTINGS_FILE = "settings/basic.yml"

# Help messages:
SETTINGS_HELP_MSG = "load settings from yaml files. " \
                    "If multiple files are specified, overlapping settings " \
                    "will be overwritten according to order of appearance " \
                    "(e.g. settings from file #1 will be overwritten by file #2)."
Q_HELP_MSG = "use n-step qlearning instead of a3c"


def _create_train_parser(description):
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--settings", "-s",
                        dest="settings_yml",
                        metavar='YAML_FILE',
                        nargs="*",
                        type=str,
                        default=[DEFAULT_SETTINGS_FILE],
                        help=SETTINGS_HELP_MSG)
    parser.add_argument("--run_tag",
                        "-rt",
                        dest="run_tag",
                        metavar="RUN_TAG",
                        help="Prefix added to tensorboard summaries",
                        default=None)
    parser.add_argument("--frameskip", "-fs",
                        dest="frameskip",
                        metavar='FRAMESKIP',
                        type=int,
                        default=None,
                        help="Override frameskip setting.")
    return parser


def _create_test_parser(description):
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(dest="model",
                        metavar="MODEL_PATH",
                        type=str,
                        help="Path to trained model directory."
                        )
    parser.add_argument("--episodes", "-e",
                        dest="episodes_num",
                        metavar="EPISODES_NUM",
                        type=int,
                        default=10,
                        help="Number of episodes to test."
                        )
    parser.add_argument("--hide-window", "-ps",
                        dest="print_settings",
                        action="store_const",
                        default=False,
                        const=True,
                        help="Hide window."
                        )
    parser.add_argument("--print-settings", "-hw",
                        dest="hide_window",
                        action="store_const",
                        default=False,
                        const=True,
                        help="Print settings upon loading."
                        )
    parser.add_argument("--fps", "-fps",
                        dest="fps",
                        metavar="FRAMERATE",
                        default=35,
                        help="If window is visible, tests will be run with given framerate."
                        )
    parser.add_argument("--agent-view",
                        "-av",
                        dest="agent_view",
                        action="store_const",
                        default=False,
                        const=True,
                        help="If True, window will display exactly what agent sees(with frameskip), "
                             "not the smoothed out version."
                        )
    parser.add_argument("--seed", "-seed",
                        dest="seed",
                        metavar="SEED",
                        default=None,
                        type=int,
                        help="Seed for ViZDoom."
                        )
    parser.add_argument("-o", "--output",
                        dest="output",
                        metavar="STATS_OUTPUT_FILE",
                        default=None,
                        help="File for output of stats"
                        )
    parser.add_argument("--deterministic", "-d",
                        dest="deterministic",
                        metavar="DETERMINISTIC",
                        default=1,
                        type=int,
                        choices=[0, 1],
                        help="If 1 dtests will be deterministic."
                        )

    return parser


def parse_train_a3c_args():
    parser = _create_train_parser(description='A3C: training script for ViZDoom.')

    return parser.parse_args()


def parse_train_adqn_args():
    parser = _create_train_parser(description='Asynchronous n-step DQN: training script for ViZDoom.')

    return parser.parse_args()


def parse_train_dqn_args():
    parser = _create_train_parser(description='DQN: training script for ViZDoom')

    return parser.parse_args()


def parse_test_dqn_args():
    parser = _create_test_parser(description='DQN: testing script for ViZDoom')

    return parser.parse_args()


def parse_test_adqn_args():
    parser = _create_test_parser(description='Asynchronous n-step DQN: testing script for ViZDoom')

    return parser.parse_args()


def parse_test_a3c_args():
    parser = _create_test_parser(description='A3C: testing script for ViZDoom')

    return parser.parse_args()
