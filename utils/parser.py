import yaml
import argparse
from argparse import Namespace
from utils.other._utils import dict_to_namespace

    
def  arg_parse():
    """
   MErge arguments from CLI and YAML

    """
    cli_args = get_cli_args()
    yml_config = load_yml_config(cli_args.config)

    merged_dict = vars(yml_config)
    merged_dict.update({k: v for k, v in vars(cli_args).items() if v is not None})
    return dict_to_namespace(merged_dict)



def get_cli_args():
    """
   Get arguments from CLI

    """
    parser = argparse.ArgumentParser(description="Argument parser for the NAS-MLP project")
    parser.add_argument("--config", type=str, default="../my_config.yml", help="Path to the config file")
    # parser.add_argument("--epochs", type=int, help="Override the number of epochs")
    return parser.parse_args()



def load_yml_config(
        config_path="my_config.yaml"
        ):
    """
    Load configuration from a YAML file.

    """
    try:
        with open(config_path, "r") as file:
            config_dict = yaml.safe_load(file)
        return Namespace(**config_dict)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {e}")
