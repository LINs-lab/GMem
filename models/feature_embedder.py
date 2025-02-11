import torch
import torch.nn as nn

class FeatureEmbedder(nn.Module):
    """
    Embeds input features into a different dimensional space and applies dropout.
    """
    def __init__(self, input_size: int, hidden_size: int, dropout_prob: float):
        """
        Initializes the FeatureEmbedder module.
        
        Args:
        - input_size (int): The size of the input feature, typically 768.
        - hidden_size (int): The size of the output embedding space.
        - dropout_prob (float): The probability of dropout to apply to the features.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        
        # Linear layer to transform input features from input_size to hidden_size
        self.linear = nn.Linear(input_size, hidden_size)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, features: torch.Tensor, train: bool) -> torch.Tensor:
        """
        Forward pass through the FeatureEmbedder module.
        
        Args:
        - features (torch.Tensor): The input features with shape (bs, z, input_size).
        - train (bool): A flag indicating whether the model is in training mode.
        
        Returns:
        - torch.Tensor: The embedded features with shape (bs, z, hidden_size).
        """
        # Apply dropout during training
        if train and self.dropout_prob > 0:
            features = self.dropout(features)
            
        # Apply the linear transformation to the input features
        embeddings = self.linear(features)
        
        return embeddings

