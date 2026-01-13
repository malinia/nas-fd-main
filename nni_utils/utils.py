from ptflops import get_model_complexity_info


def compute_objective(model, auc, opt_mode="flops", input_size=2, max_flops_size=134145, alpha=0.9, beta=0.1):

   
    input_shape = (input_size,)  

    # Compute FLOPs and parameters
    # with torch.cuda.device(0):  # If using CUDA; omit if using CPU
    flops, params = get_model_complexity_info(model, input_shape, as_strings=False, print_per_layer_stat=False)
    total_params = sum(p.numel() for p in model.parameters())

    print("----------------------------- Objective Function -----------------------------\n")
    print(f"Alpha : {alpha}")
    print(f"Beta : {beta}")
    print("FLOPs:", flops)
    print("Parameters:", params)
    print("Parameters 2 :", total_params)

    if opt_mode=="flops":
        normalized_model_spec = flops / max_flops_size
    elif (opt_mode=="size"):
        normalized_model_spec = total_params / max_flops_size
    print(f"normalized {opt_mode}: {normalized_model_spec}")
    print(f"AUC: {auc} \n \n \n")
    
    # Calculate objective as a weighted sum
    objective_value = alpha * auc - beta * normalized_model_spec

    return objective_value, flops, total_params