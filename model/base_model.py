import torch
import torch.nn as nn
import torch.optim as optim
import joblib


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def forward(self, x):
        return self.model(x)

    def save_model(self, path):
        """  Save the model's state_dict to the specified path.  """
        # torch.save(self.state_dict(), path)
        joblib.dump(self.state_dict(), f"{path}model.joblib") 
        print(f"Model saved to {path}")


    def load_model(self, path):
        """  Load the model's state_dict from the specified path.  """
        self.load_state_dict(torch.load(path))



class MLPModel(BaseModel):
    """ MLP models"""
    def __init__(self,
                 input_size:int,
                 hidden_layer_1_neurons:int, 
                 hidden_layer_2_neurons:int, 
                 hidden_layer_3_neurons:int, 
                 hidden_layer_4_neurons:int, 
                 output_size:int
                 ):
        super(MLPModel, self).__init__()

        layers = []
        layers.append(nn.Linear(input_size, hidden_layer_1_neurons))
        layers.append(nn.ReLU())
        
        previous_neurons = hidden_layer_1_neurons
        
        # Add second layer if it has neurons
        if hidden_layer_2_neurons > 0:
            layers.append(nn.Linear(hidden_layer_1_neurons, hidden_layer_2_neurons))
            layers.append(nn.ReLU())
            previous_neurons = hidden_layer_2_neurons
        
        # Add third layer if it has neurons
        if hidden_layer_3_neurons > 0:
            layers.append(nn.Linear(previous_neurons, hidden_layer_3_neurons))
            layers.append(nn.ReLU())
            previous_neurons = hidden_layer_3_neurons

        # Add fourth layer if it has neurons
        if hidden_layer_4_neurons > 0:
            layers.append(nn.Linear(previous_neurons, hidden_layer_4_neurons))
            layers.append(nn.ReLU())
            previous_neurons = hidden_layer_4_neurons

        # Output layer
        layers.append(nn.Linear(previous_neurons, output_size))
        # layers.append(nn.Sigmoid())

        # Assign layers to Sequential
        self.model = nn.Sequential(*layers)

        self.model.apply(self._initialize_weights)
        

    def _initialize_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

                
    
class CNNModel(BaseModel):
    """
    ...
    """
    #TODO add the other models