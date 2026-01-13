from argparse import Namespace
import hashlib
import time
import os
from datetime import datetime
import pickle
import yaml
import joblib
import logging
import sys


def dict_to_namespace(d):
    """
    ecursively convert dictionaries to Namespace
    """
    if isinstance(d, dict):
        return Namespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(i) for i in d]
    else:
        return d
    

def args_assess(my_args):
    """
    assess arguments 
    """
    if not isinstance(my_args, dict) and not isinstance(my_args, Namespace):
        raise ValueError("Arguments must be a dictionary or a Namespace object.")
    
    if not isinstance(my_args.mode, str) :
        raise ValueError("Config eroor: Argument \"mode\" must be an str object.")
    assert my_args.mode in ['manualm', 'nnim'], "Argument \"mode\" must set to  manualm or nnim"


    if not isinstance(my_args.training.epochs, int) :
        raise ValueError("Config eroor: Argument \"training.epochs\" must be an int object.")
    assert 1 <= my_args.training.epochs <= 2000 , "Argument \"training.epochs\" is too small or too larg"

    #TODO continue the assessmnet of the other args
    # if not isinstance(my_args.training.epochs, int) :
    #     raise ValueError("Config eroor: Argument \"training.epochs\" must be an int object.")
    # assert my_args.training.epochs>1 and my_args.training.epochs>2000 , "Argument \"training.epochs\" is too small or too larg"



def prepare_experiment(mode, 
                       data_type, 
                       output_dir
                       ):

    if mode=="nnim":
        output_dir2= output_dir+"nni/"
    elif mode=="manualm":
        output_dir2=output_dir+data_type+"/"

     # Use the current timestamp and hash it
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    timestamp1 = str(time.time()).encode()
    run_id = hashlib.sha256(timestamp1).hexdigest()[:10]  # First 10 characters
    # print(f"Run ID:{timestamp1}", run_id)

    experiment_dir=f"{output_dir2}{timestamp}__{run_id}"
    try:
        os.makedirs(experiment_dir, exist_ok=True)
        print(f"Directory '{experiment_dir}' created successfully")
    except Exception as e:
        print(f"Error creating directory: {e}") 

    return timestamp, run_id, experiment_dir+"/"



def save_experiment(object, 
                    to_save:str, 
                    experiment_dir,
                    timestamp,
                    run_id 
                    ):
    
    assert to_save in ["model", "config", "result"], "save_experiment, to_save must be in ['model', 'config', 'result']"

    if to_save == "model":
        model_dict = object.state_dict()
        joblib.dump(model_dict, f"{experiment_dir}{timestamp}_model_{run_id}.joblib")

    elif to_save == "config":
        with open(f'{experiment_dir}experiment_config.pkl', 'wb') as file:
            pickle.dump(object, file)
        with open(f'{experiment_dir}experiment_config.yml', 'w') as file:
         yaml.dump(vars(object), file, default_flow_style=False)

    elif to_save == "result":
        pass



def setup_logging(output_dir, 
                  log_filename
                  ):
    
    # Define log file path
    log_file_path = os.path.join(output_dir, log_filename)
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,  # Log level: DEBUG, INFO, WARNING, etc.
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file_path),  # Log to file
            logging.StreamHandler(sys.stdout)  # Also log to console
        ]
    )
    
    # Redirect stdout and stderr to the log file
    sys.stdout = open(log_file_path, "a")  # Append mode
    sys.stderr = sys.stdout  # Redirect errors to the same file
    
    logging.info("Logging initialized. All output will be written to the log file.")



