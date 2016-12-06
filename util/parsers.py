import argparse


def parse_train_a3c_args():
    parser = argparse.ArgumentParser(description='A3c implementation for ViZDoom in Tensorflow.')
    parser.add_argument("--yaml", "-y", dest="settings_yml", metavar='<SETTINGS_YAML_FILES>',
                        default=["settings/basic.yml"],
                        type=str, required=False, help="paths of yamls with settings")
    # TODO run
    # TODO tag
    # TODO tags extension

    return parser.parse_args()


def parse_train_dqn_args():
    parser = argparse.ArgumentParser(description='DQN implementation for ViZDoom in Tensorflow.')
    parser.add_argument("--yaml", "-y", dest="settings_yml", metavar='<SETTINGS_YAML_FILES>',
                        default=["settings/basic.yml"],
                        type=str, required=False, help="paths of yamls with settings")
    # TODO run
    # TODO tag
    # TODO tags extension

    return parser.parse_args()

