from utils.parser import arg_parse
from utils.other._utils import *
from utils.read_data import *
from utils.prepare_data import *
from model.base_model import *
from model.model_utils import *
import nni 
from nni import get_next_parameter


def main():
    # get the config args
    config=arg_parse()
    #generate a unique ID for the experiment if not based on nni
    # print(config.training.epochs)
    args_assess(config)

    timestamp, run_id, experiment_dir = prepare_experiment(config.mode, config.data.data_type, config.output_dir)
    setattr(config, 'timestamp', timestamp)
    setattr(config, 'run_id', run_id)
    setattr(config, 'experiment_dir', experiment_dir)
    
    setup_logging(config.experiment_dir, log_filename="experiment_output.log")
    print(f"Experiment dir: {config.experiment_dir}")
    print(f"Experiment timestamp: {config.timestamp}")
    print(f"Experiment id: {config.run_id}")

    # Check if CUDA is available and select the appropriate device
    device = torch.device(config.device)
    print(f"Device in use: {device}")

    save_experiment(config, "config", config.experiment_dir, config.timestamp, config.run_id)

    Xtrain, Xtest, Ytrain, Ytest = read_data(config.data.data_path)

    train_loader, test_loader, val_loader = get_data_loader(config.data.data_type, Xtrain, Xtest, Ytrain, Ytest, config.training.batch_size)
    print(type(train_loader))


    if config.data.data_type=="euc" or "dtw":
        input_size = 2
    else:
        pass
    
    params = None
    if config.mode == "nnim":
        params = get_next_parameter() # Load parameters from NNI
        lr_rate = params["learning_rate"]
    elif config.mode == "manualm":
        lr_rate = config.training.learning_rate

    output_size=1

    if config.ml_model.model_type == "mlp":
       model = build_model(input_size, output_size, params, config, model=MLPModel)
       
    print_summary(model)

    pos_weight = get_class_weights(train_loader)
    pos_weight = pos_weight.to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.Adam(model.parameters(), lr=lr_rate, weight_decay=1e-4)
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
   
    aucs, _, _ = train_model(model, 
          train_loader,
          val_loader, 
          criterion, 
          optimizer, 
          scheduler, 
          config.training.epochs, 
          device,
          config.mode)
    
    metric_auc_flops, flops, total_params = compute_objective(model, aucs[-1])

    if config.mode=="nnim":
        nni.report_final_result(metric_auc_flops)

    metrics, _, _=evaluate_model(model,
                   test_loader, 
                   " test data ",
                   device)
    
    # To check memory allocated
    print(f"Allocated memory : {torch.cuda.memory_allocated()}")  

    # To check memory cached
    print(f"Reserved memory : {torch.cuda.memory_reserved()}") 

    cpu_time_single = calculate_inference_time(model, test_loader, single_instance=True, device="cpu")
    # time_full = calculate_inference_time(model, dataset, single_instance=False, device="cuda")
    gpu_time_single = calculate_gpu_inference_time(model, test_loader, single_instance=True, device="cuda")
    # gpu_time_full = calculate_gpu_inference_time(model, dataset, single_instance=False, device="cuda")
    # TODO save metrics and model
    # TODO check if test data was normalized
    # save results
    evaluation_results = {
    "Timestamp": [config.timestamp],
    "Experiment_ID": [config.run_id],
    "Experiment_dir": [config.experiment_dir],
    "Distance": [config.data.data_type],
    "Neurons": [config.ml_model.neurons_number],
    "Epochs": [config.training.epochs],
    "Accuracy": [metrics['accuracy']],
    "AUC": [metrics['auc']],
    "TPR": [metrics['tpr']],
    "FPR": [metrics['fpr']],
    "Flops": [flops],
    "NumParams": [total_params],
    "cpu_time_single": [cpu_time_single],
    "gpu_time_single": [gpu_time_single]
    }

    # save model and results
    model.save_model(config.experiment_dir)
    pd.DataFrame(evaluation_results).to_csv(f"{config.experiment_dir}results.csv")


if __name__=="__main__":
    main()