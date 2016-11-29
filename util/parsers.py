import argparse


def parse_train_a3c_args():
    parser = argparse.ArgumentParser(description='A3c implementation for ViZDoom in Tensorflow.')
    parser.add_argument("--json", "-j", dest="settings_json", metavar='<SETTINGS_JSON_FILE>',
                        default="settings/a3c/basic.json",
                        type=str, required=False, help="path to json with settings")
    # TODO run
    # TODO tag
    # TODO tags extension

    return parser.parse_args()


def parse_train_dqn_args():
    parser = argparse.ArgumentParser(description='DQN implementation for ViZDoom in Tensorflow.')
    parser.add_argument("--json", "-j", dest="settings_json", metavar='<SETTINGS_JSON_FILE>',
                        default="settings/dqn/basic.json",
                        type=str, required=False, help="path to json with settings")
    # TODO run
    # TODO tag
    # TODO tags extension

    return parser.parse_args()

