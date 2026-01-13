
import time
import matplotlib.pyplot as plt
import torch 
from model.base_model import *
from nni_utils.utils import compute_objective
from torchinfo import summary
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import auc, roc_curve, confusion_matrix
# from tqdm import tqdm
import nni
from collections import Counter


def build_model(input_size,
                output_size,
                params, 
                config,
                model=MLPModel
                ):
    
    if config.mode == "nnim":
        #for now up to three layers
        return model(
        input_size,
        params["hidden_layer_1_neurons"],
        params["hidden_layer_2_neurons"],
        params["hidden_layer_3_neurons"],
        0,
        output_size)
    
    elif config.mode == "manualm":
        assert config.ml_model.architecture in ['pyramid', 'inversepyramid', 'sandwich', 'hourglass', 'uniform'], " config.ml_model.architecture should be equal to 'pyramid', 'inversepyramid', 'sandwich' or 'hourglass'"

        if config.ml_model.architecture == 'inversepyramid':
            return model(
            input_size,
            config.ml_model.neurons_number,
            config.ml_model.neurons_number*2,
            config.ml_model.neurons_number*4,
            config.ml_model.neurons_number*8,
            output_size
        )
        elif config.ml_model.architecture == 'pyramid':
            return model(
            input_size,
            config.ml_model.neurons_number,
            config.ml_model.neurons_number//2,
            config.ml_model.neurons_number//4,
            config.ml_model.neurons_number//8,
            output_size
        )
        elif config.ml_model.architecture == 'sandwich':
            return model(
            input_size,
            config.ml_model.neurons_number,
            config.ml_model.neurons_number*2,
            config.ml_model.neurons_number,
            0,
            output_size
        )
        elif config.ml_model.architecture == 'hourglass':
            return model(
            input_size,
            config.ml_model.neurons_number,
            config.ml_model.neurons_number//2,
            config.ml_model.neurons_number,
            0,
            output_size
            )
        elif config.ml_model.architecture == 'uniform':
            return model(
            input_size,
            config.ml_model.neurons_number,
            config.ml_model.neurons_number,
            config.ml_model.neurons_number,
            0,
            output_size
            )

    

# # Define the training function
# def train_model(model, 
#           train_loader,
#           val_loader, 
#           criterion, 
#           optimizer, 
#           scheduler, 
#           epochs, 
#           device
#           ):
    

#     train_losses = []  # To store training losses for each epoch
#     grad_norms = []         # Store average gradient norms per epoch
#     aucs=[]

#     for epoch in range(epochs):
#         model.train() #training mode
#         epoch_loss = 0  # Initialize epoch loss
#         epoch_grad_norms = []  # Store gradient norms for this epoch

#         for batch_idx, (data, target) in enumerate(train_loader):
#             # print(f"data: {data}")
#             # print(f"target: {target}")
#             data, target = data.to(device), target.to(device)
#             optimizer.zero_grad()
#             output = model(data.view(data.size(0), -1))
#             loss = criterion(output, target)
#             loss.backward()

#             # Save gradient norms for this batch
#             total_grad_norm = 0
#             for name, param in model.named_parameters():
#                 if param.grad is not None:
#                     grad_norm = param.grad.norm(2).item()  # L2 norm of gradients
#                     total_grad_norm += grad_norm
#             epoch_grad_norms.append(total_grad_norm)

#             optimizer.step()

#             # Accumulate batch loss
#             epoch_loss += loss.item()

            

#         # Update learning rate
#         scheduler.step()    
        
#          # Update weight decay
#         if epoch % 10 == 0:  # Adjust weight decay every 10 epochs
#             adjust_weight_decay(optimizer, epoch, decay_rate=0.9)
        
#         # Average loss for the epoch
#         epoch_loss /= len(train_loader)
#         train_losses.append(epoch_loss)
#         avg_grad_norm = sum(epoch_grad_norms) / len(epoch_grad_norms)
#         grad_norms.append(avg_grad_norm)

#         print(f"************************************* Epoch {epoch+1}/{epochs} *************************************\n Loss: {epoch_loss:.4f}, Avg Grad Norm: {avg_grad_norm:.4f}")



#             # Report accuracy to NNI
#         nni.report_intermediate_result(auc)

#     metric_auc_flops, flops, total_params=compute_objective(model, auc)
#     # nni.report_final_result(auc)  
#     nni.report_final_result(metric_auc_flops)  

#     # save the model  
#     # Convert the model’s state_dict to a dictionary and save with joblib
#     timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#     model_dict = model.state_dict()
#     joblib.dump(model_dict, f"models/DTW2/{timestamp}_model_{nni.get_experiment_id()}_{nni.get_trial_id()}.joblib") 




# Define the training function
def train_model(model,
                train_loader,
                val_loader,
                criterion,
                optimizer,
                scheduler,
                epochs,
                device,
                mode
                ):
    

    train_losses = []  
    grad_norms = []    
    aucs=[]

    for epoch in range(epochs):
        epoch_loss = 0  
        epoch_grad_norms = []  
        model.train() 
       
        for batch_idx, (data, target) in enumerate(train_loader):
            # print(f"data: {data}")
            # print(f"target: {target}")
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data.view(data.size(0), -1))
            loss = criterion(output, target)
            loss.backward()

            # Save gradient norms for this batch
            total_grad_norm = 0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm(2).item()
                    total_grad_norm += grad_norm
            epoch_grad_norms.append(total_grad_norm)

            optimizer.step()

            
            epoch_loss += loss.item()


        # Update learning rate
        scheduler.step()    
        
         # Update weight decay
        if epoch % 10 == 0:  
            adjust_weight_decay(optimizer, epoch, decay_rate=0.9)
        
        # Average loss for the epoch
        epoch_loss /= len(train_loader)
        train_losses.append(epoch_loss)
        avg_grad_norm = sum(epoch_grad_norms) / len(epoch_grad_norms)
        grad_norms.append(avg_grad_norm)

        print(f"************************************* Epoch {epoch+1}/{epochs} *************************************\n Loss: {epoch_loss:.4f}, Avg Grad Norm: {avg_grad_norm:.4f}")


        metrics, preds, labels = evaluate_model(model, val_loader, " validation data ", device)
        aucs.append(metrics['auc'])

        if mode=="nnim":

            nni.report_intermediate_result(aucs[-1])

    

    
    return aucs, train_losses, grad_norms



def adjust_weight_decay(optimizer, 
                        epoch, 
                        decay_rate=0.9
                        ):
    """Custom weight decay schedule"""
    for param_group in optimizer.param_groups:
        param_group['weight_decay'] *= decay_rate



def evaluate_model(model,
                      test_loader, 
                      test_data_name,
                      device
                      ):

    model.eval()

    with torch.no_grad():
        all_preds = []
        all_labels = []
        
        for inputs, labels in test_loader:
            # Forward pass: Get model outputs (predictions)
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
    #         # Convert the outputs to binary predictions (e.g., 0 or 1 for binary classification)
    #         predicted = (outputs > 0.5).float()  # For binary classification
            
    #         all_predictions.append(predicted)
    #         all_labels.append(targets)

    # # Convert the list of predictions and labels to tensors for further evaluation
    # all_predictions = torch.cat(all_predictions, dim=0)
    # all_labels = torch.cat(all_labels, dim=0)

    # # Calculate accuracy
    # accuracy = (all_predictions == all_labels).float().mean()
    # return accuracy.item()

            probs = torch.sigmoid(outputs).squeeze()  # Get probabilities
            all_labels.extend(labels.cpu().numpy())  # True labels
            all_preds.extend(probs.cpu().numpy())  # Predicted probabilities

   
        # Convert probabilities to binary predictions
    all_preds_binary = [1 if p > 0.5 else 0 for p in all_preds]

    # Calculate metrics
    metrics = calculate_metrics(all_labels, all_preds_binary)
       

    # Print metrics for debugging or logging
    if test_data_name==" test data " or " validation data ":
        print(f"*********** Results of inference on {test_data_name} ********")
        print(f" > AUC: {metrics['auc']}")
        # auc_euc_mlp = auc(f1_euc_mlp, t1_euc_mlp)
        # print(f" > TPR: {metrics["tpr"]}, FPR: {metrics["fpr"]}")
        print(f" > TPR: {metrics['tpr']}")
        print(f" > FPR: {metrics['fpr']}")
        print(f" > Accuracy: {metrics['accuracy']}")
        print(f" > Precision: {metrics['precision']}")
        print(f" > Recall: {metrics['recall']}")
        print(f" > F1 Score: {metrics['f1']} \n \n ")


    # Report all metrics as a dictionary to NNI (either intermediate or final result)
    # metrics = {
    #     'AUC': auc,
    #     'Accuracy': accuracy,
    #     'Precision': precision,
    #     'Recall': recall,
    #     'F1 Score': f1
    # }

    # Report the metrics (you can use either intermediate or final result depending on the stage)
    # nni.report_intermediate_result(metrics)  # Report intermediate metrics (if in training loop)
    
    # Return the metrics if needed
    return metrics, all_preds_binary, all_labels



def calculate_metrics(all_labels, all_preds_binary):
    auc = roc_auc_score(all_labels, all_preds_binary)
    accuracy = accuracy_score(all_labels, all_preds_binary)
    precision = precision_score(all_labels, all_preds_binary, zero_division=0)
    recall = recall_score(all_labels, all_preds_binary)
    f1 = f1_score(all_labels, all_preds_binary)

    # f1_euc_mlp, t1_euc_mlp, th1=roc_curve(all_labels, all_preds_binary, pos_label=1)
      
    CM = confusion_matrix(all_labels, all_preds_binary)

    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
        
    tpr=(TP/(TP))
    fpr=(FP/(FP)) 

    metrics={"auc": auc,
             "accuracy": accuracy,
             "precision": precision,
             "recall": recall,
             "f1": f1,
             "TN": TN,
             "FP": FP,
             "FN": FN,
             "TP": TP,
             "tpr": tpr,
             "fpr": fpr
             }
    return metrics



def calculate_inference_time(model,
                            dataset,
                            single_instance=False, 
                            device="cpu"
                            ):
    """
    Calculates the inference time of a PyTorch model.

    Parameters:
        model (torch.nn.Module): The PyTorch model to evaluate.
        dataset (torch.utils.data.Dataset or torch.Tensor): The dataset or tensor for inference.
        single_instance (bool): If True, calculates the time for one instance only. Defaults to False (whole dataset).
        device (str): The device to run the model on ('cpu' or 'cuda'). Defaults to 'cpu'.

    Returns:
        float: Inference time in seconds.
    """
    model = model.to(device)
    model.eval()
    data_iter = iter(dataset)
    # # Prepare data for inference
    # if isinstance(dataset, torch.utils.data.Dataset):
    #     dataloader = torch.utils.data.DataLoader(dataset, batch_size=1 if single_instance else len(dataset))
    #     data_iter = iter(dataloader)
    # elif isinstance(dataset, torch.Tensor):
    #     if single_instance:
    #         dataset = dataset[:1]
    #     dataloader = [(dataset.to(device),)]
    #     data_iter = iter(dataloader)
    # else:
    #     raise ValueError("Unsupported dataset type. Must be a PyTorch Dataset or Tensor.")

    # Measure inference time
    with torch.no_grad():
        start_time = time.perf_counter() #time()
        for inputs in data_iter:
            inputs = inputs[0].to(device)  # Assuming dataset or dataloader provides inputs in first position
            _ = model(inputs)
            if single_instance:
                break
        end_time = time.perf_counter() #time()

    return end_time - start_time



def calculate_gpu_inference_time(model,
                                dataset, 
                                single_instance=False,
                                device="cuda"
                                ):
    """
    Calculates the inference time of a PyTorch model using CUDA events for precise GPU timing.

    Parameters:
        model (torch.nn.Module): The PyTorch model to evaluate.
        dataset (torch.utils.data.Dataset or torch.Tensor): The dataset or tensor for inference.
        single_instance (bool): If True, calculates the time for one instance only. Defaults to False (whole dataset).
        device (str): The device to run the model on ('cuda'). Defaults to 'cuda'.

    Returns:
        float: Inference time in seconds.
    """
    if device != "cuda":
        raise ValueError("CUDA inference requires device to be 'cuda'.")

    model = model.to(device)
    model.eval()
    data_iter = iter(dataset)
    # Prepare data for inference
    # if isinstance(dataset, torch.utils.data.Dataset):
    #     dataloader = torch.utils.data.DataLoader(dataset, batch_size=1 if single_instance else len(dataset))
    #     data_iter = iter(dataloader)
    # elif isinstance(dataset, torch.Tensor):
    #     if single_instance:
    #         dataset = dataset[:1]
    #     dataloader = [(dataset.to(device),)]
    #     data_iter = iter(dataloader)
    # else:
    #     raise ValueError("Unsupported dataset type. Must be a PyTorch Dataset or Tensor.")

    # Measure inference time using CUDA events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        start_event.record()
        for inputs in data_iter:
            inputs = inputs[0].to(device)  # Assuming dataset or dataloader provides inputs in first position
            _ = model(inputs)
            if single_instance:
                break
        end_event.record()

    # Wait for all GPU tasks to complete
    torch.cuda.synchronize()
    inference_time = start_event.elapsed_time(end_event) / 1000  # Convert from ms to seconds

    return inference_time



def plot_results(epochs,
                 train_losses, 
                 grad_norms, 
                 auc_values, 
                 path):
    

    plt.figure(figsize=(15, 5))  # Increase figure width for an extra subplot

    # Plot loss curve
    plt.subplot(1, 3, 1)
    plt.plot(range(1, epochs + 1), train_losses, label="Training Loss", marker='o')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True)

    # Plot gradient norms
    plt.subplot(1, 3, 2)
    plt.plot(range(1, epochs + 1), grad_norms, label="Avg Gradient Norm", marker='o', color='orange')
    plt.xlabel("Epochs")
    plt.ylabel("Gradient Norm")
    plt.title("Gradient Norms Over Epochs")
    plt.legend()
    plt.grid(True)

    # Plot AUC curve
    plt.subplot(1, 3, 3)
    plt.plot(range(1, epochs + 1), auc_values, label="AUC", marker='o', color='green')
    plt.xlabel("Epochs")
    plt.ylabel("AUC")
    plt.title("AUC Over Epochs")
    plt.legend()
    plt.grid(True)

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    plt.savefig(path+"training_metrics.png")



def print_summary(model):

    print("\n-------------- Model Summary -------------- \n")
    summary(model, input_size=(2,))
    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            num_neurons = layer.out_features if hasattr(layer, 'out_features') else layer.out_channels
            print(f"Layer: {name}, Type: {layer.__class__.__name__}, Neurons: {num_neurons}")

    # Print model weights
    # for name, param in model.named_parameters():
    #     print(f"Layer: {name} | Shape: {param.shape}")
    #     print(param)  # Prints the actual weight tensor

    return None


def get_class_weights(train_loader):
    # Calculate class counts from the DataLoader
    class_counts = Counter()
    for _, targets in train_loader:
        class_counts.update(targets.view(-1).tolist())
    # Calculate pos_weight
    pos_weight = torch.tensor([class_counts[0]/ class_counts[1]]) #num_negative / num_positive    
    print(f"****** num_negative : {class_counts[0]}, num_positive : {class_counts[1]}")
    return pos_weight