import argparse


def parse_train_async_args():
    parser = argparse.ArgumentParser(description='A3c implementation for ViZDoom in Tensorflow.')
    parser.add_argument("-s", "--settings", dest="settings_yml", metavar='<SETTINGS_YAML_FILES>',
                        default=["settings/basic.yml"], nargs="*",
                        type=str, required=False, help="paths of yamls with settings")
    parser.add_argument("-q", action="store_const", default=False, const=True,
                        help="use n-step qlearning instead of a3c", dest="q")
    # TODO run
    # TODO tag
    # TODO tags extension

    return parser.parse_args()


def parse_train_dqn_args():
    parser = argparse.ArgumentParser(description='DQN implementation for ViZDoom in Tensorflow.')
    parser.add_argument("-s", "--settings", dest="settings_yml", metavar='<SETTINGS_YAML_FILES>',
                        default=["settings/basic.yml"], nargs="*",
                        type=str, required=False, help="paths of yamls with settings")
    # TODO run
    # TODO tag
    # TODO tags extension

    return parser.parse_args()
