import torch

import torch
from abc import ABC, abstractmethod

class BaseMemoryBank(ABC):
    """
    Abstract base class for memory bank implementations.
    
    Attributes:
        capacity: Current number of stored memory snippets (read-only)
        feature_dim: Original feature dimension (read-only)
    """
    
    @property
    @abstractmethod
    def capacity(self) -> int:
        """Current number of stored memory snippets"""
        pass

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """Original feature dimension of memory snippets"""
        pass

    @abstractmethod
    def get(self, index: int) -> torch.Tensor:
        """
        Retrieve a memory snippet by index.
        
        Args:
            index: Position in memory bank (0 <= index < capacity)
            
        Returns:
            Memory snippet tensor [feature_dim]
        """
        pass

    @abstractmethod
    def add_exact(self, snippet: torch.Tensor) -> torch.Tensor:
        """
        Add a new snippet to the memory bank.
        
        Args:
            snippet: New feature vector [feature_dim]
            
        Returns:
            Added snippet [feature_dim]
        """
        pass

    @abstractmethod
    def add_interpolated(self, i: int, j: int, alpha: float = 0.5) -> torch.Tensor:
        """
        Generate new snippet by interpolating between existing entries.
        
        Args:
            i: Index of first base snippet
            j: Index of second base snippet
            alpha: Interpolation weight (0.0-1.0)
            
        Returns:
            New interpolated snippet [feature_dim]
        """
        pass
    
    @abstractmethod
    def get_interpolated(self, i: int, j: int, alpha: float = 0.5) -> torch.Tensor:
        """
        Generate new snippet by interpolating between existing entries.
        
        Args:
            i: Index of first base snippet
            j: Index of second base snippet
            alpha: Interpolation weight (0.0-1.0)
            
        Returns:
            New interpolated snippet [feature_dim]
        """
        pass
    
    @abstractmethod
    def get_exact(self, snippet: torch.Tensor) -> torch.Tensor:
        """
        Add a new snippet to the memory bank.
        
        Args:
            snippet: New feature vector [feature_dim]
            
        Returns:
            Added snippet [feature_dim]
        """
        pass

    def __getitem__(self, index: int) -> torch.Tensor:
        """Enable direct indexing (memory_bank[i])"""
        return self.get(index)

    def __len__(self) -> int:
        """Return current memory capacity"""
        return self.capacity
    
    def to(self, device):
        """
        Move memory bank to a new device.
        
        Args:
            device: Target device (e.g. 'cpu', 'cuda:0')
            
        Returns:
            Memory bank instance with updated device
        """
        return self
    
class FullMemoryBank(BaseMemoryBank):
    """
    Memory bank that stores snippets in their original form (no compression).
    """
    
    def __init__(self, bank: torch.Tensor):
        """
        Initialize memory bank with existing snippets.
        
        Args:
            bank: Initial memory snippets [N, feature_dim]
        """
        self._bank = bank.clone()
        self._feature_dim = bank.shape[1]

    @classmethod
    def from_state_dict(cls, path: str, device: str='cpu'):
        """
        Load memory bank from saved state dictionary.
        
        Args:
            state_dict: Dictionary containing:
                - 'C': Coefficient matrix [N, d]
                - 'M': Basis matrix [feature_dim, d]
                - 'S': Singular values [d]
                - 'mean': Mean vector [feature_dim]
                - 'latent_dim': Latent dimension
                - 'feature_dim': Feature dimension
        """
        instance = cls.__new__(cls)
        state_dict = torch.load(path, map_location=device)
        assert type(state_dict) == torch.Tensor
        
        instance._bank = state_dict
        instance._feature_dim = state_dict.shape[1]
        return instance

    def save(self, file_path: str):
        """
        Save current state to file.
        
        Args:
            file_path: Path to save the state dictionary
        """
        state_dict = self._bank
        torch.save(state_dict, file_path)

    @property
    def capacity(self) -> int:
        return len(self._bank)

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def get(self, index: int) -> torch.Tensor:
        self._validate_index(index)
        return self._bank[index]

    def add_exact(self, snippet: torch.Tensor) -> torch.Tensor:
        if snippet.shape != (self.feature_dim,):
            raise ValueError(f"Snippet must have shape ({self.feature_dim},)")
        self._bank = torch.cat([self._bank, snippet.unsqueeze(0)], dim=0)
        return self.get(-1)

    def add_interpolated(self, i: int, j: int, alpha: float = 0.5) -> torch.Tensor:
        self._validate_index(i)
        self._validate_index(j)
        interp_snippet = alpha * self._bank[i] + (1 - alpha) * self._bank[j]
        return self.add_exact(interp_snippet)
    
    def get_interpolated(self, i: int, j: int, alpha: float = 0.5) -> torch.Tensor:
        self._validate_index(i)
        self._validate_index(j)
        return alpha * self._bank[i] + (1 - alpha) * self._bank[j]
    
    def get_exact(self, snippet: torch.Tensor) -> torch.Tensor:
        if snippet.shape != (self.feature_dim,):
            raise ValueError(f"Snippet must have shape ({self.feature_dim},)")
        return snippet

    def _validate_index(self, index: int):
    #     if not index < self.capacity:
    #         raise IndexError
        pass
        
    def to(self, device):
        self._bank = self._bank.to(device)
        return self
        
        

class DMDMemoryBank(BaseMemoryBank):
    """
    A compressed memory bank using matrix decomposition with interactive operations.
    
    Attributes:
        capacity: Current number of stored memory snippets (read-only)
        feature_dim: Original feature dimension (read-only)
        latent_dim: Compressed latent dimension (read-only)
    """
    
    def __init__(self, bank: torch.Tensor, latent_dim: int = 64):
        """
        Initialize memory bank with existing snippets.
        
        Args:
            bank: Initial memory snippets [N, feature_dim]
            latent_dim: Target latent dimension for compression
        """
        self._initialize_from_data(bank, latent_dim)

    @classmethod
    def from_state_dict(cls, path: str):
        """
        Load memory bank from saved state dictionary.
        
        Args:
            state_dict: Dictionary containing:
                - 'C': Coefficient matrix [N, d]
                - 'M': Basis matrix [feature_dim, d]
                - 'S': Singular values [d]
                - 'mean': Mean vector [feature_dim]
                - 'latent_dim': Latent dimension
                - 'feature_dim': Feature dimension
        """
        instance = cls.__new__(cls)
        state_dict = torch.load(path)
        assert type(state_dict) == dict
        instance._C = state_dict['C']
        instance._M = state_dict['M']
        instance._S = state_dict['S']
        instance._mean = state_dict['mean']
        instance._latent_dim = state_dict['latent_dim']
        instance._feature_dim = state_dict['feature_dim']
        return instance

    def save(self, file_path: str):
        """
        Save current state to file.
        
        Args:
            file_path: Path to save the state dictionary
        """
        state_dict = {
            'C': self._C,
            'M': self._M,
            'S': self._S,
            'mean': self._mean,
            'latent_dim': self._latent_dim,
            'feature_dim': self._feature_dim
        }
        torch.save(state_dict, file_path)
        
    def feature_dim(self) -> int:
        """Original feature dimension of memory snippets"""
        return self._feature_dim

    def _initialize_from_data(self, bank: torch.Tensor, latent_dim: int):
        """Internal initialization from raw data"""
        # Center data and perform SVD
        self._mean = bank.mean(dim=0)
        centered = bank - self._mean
        U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
        
        # Truncate to latent dimension
        self._S = S[:latent_dim]
        self._C = (U[:, :latent_dim] * torch.sqrt(self._S.unsqueeze(0))).clone()
        self._M = (Vh.T[:, :latent_dim] * torch.sqrt(self._S.unsqueeze(0))).clone()
        
        # Public attributes
        self._latent_dim = latent_dim
        self._feature_dim = bank.shape[1]

    @property
    def capacity(self) -> int:
        """Current number of stored memory snippets"""
        return self._C.shape[0]

    def get(self, index: int) -> torch.Tensor:
        """
        Retrieve a memory snippet by index.
        
        Args:
            index: Position in memory bank (0 <= index < capacity)
            
        Returns:
            Decompressed snippet tensor [feature_dim]
        """
        self._validate_index(index)
        return self._C[index] @ self._M.T + self._mean

    def add_exact(self, snippet: torch.Tensor) -> torch.Tensor:
        """
        Add a new snippet through exact projection.
        
        Args:
            snippet: New feature vector [feature_dim]
            
        Returns:
            Added snippet in decompressed form [feature_dim]
        """
        # Project to latent space
        centered = snippet - self._mean
        coeff = (centered @ self._M) / self._S
        
        # Update storage
        self._C = torch.cat([self._C, coeff.unsqueeze(0)], dim=0)
        return self.get(-1)

    def add_interpolated(self, i: int, j: int, alpha: float = 0.5) -> torch.Tensor:
        """
        Generate new snippet by interpolating between existing entries.
        
        Args:
            i: Index of first base snippet
            j: Index of second base snippet
            alpha: Interpolation weight (0.0-1.0)
            
        Returns:
            New interpolated snippet [feature_dim]
        """
        self._validate_index(i)
        self._validate_index(j)
        
        # Linear interpolation in latent space
        interp_coeff = alpha * self._C[i] + (1 - alpha) * self._C[j]
        self._C = torch.cat([self._C, interp_coeff.unsqueeze(0)], dim=0)
        return self.get(-1)
    
    def get_interpolated(self, i: int, j: int, alpha: float = 0.5) -> torch.Tensor:
        """
        Generate new snippet by interpolating between existing entries.
        
        Args:
            i: Index of first base snippet
            j: Index of second base snippet
            alpha: Interpolation weight (0.0-1.0)
            
        Returns:
            New interpolated snippet [feature_dim]
        """
        self._validate_index(i)
        self._validate_index(j)
        return alpha * self._C[i] + (1 - alpha) * self._C[j]
    
    def get_exact(self, snippet: torch.Tensor) -> torch.Tensor:
        """
        Add a new snippet through exact projection.
        
        Args:
            snippet: New feature vector [feature_dim]
            
        Returns:
            Added snippet in decompressed form [feature_dim]
        """
        return snippet

    def _validate_index(self, index: int):
        """Ensure index is within valid range"""
        # if not index < self.capacity:
        #     raise IndexError(f"Index {index} out of range [0, {self.capacity-1}]")
        pass

    def __getitem__(self, index: int) -> torch.Tensor:
        """Enable direct indexing (memory_bank[i])"""
        return self.get(index)

    def __len__(self) -> int:
        """Return current memory capacity"""
        return self.capacity
    
    def to(self, device):
        """
        Move memory bank to a new device.
        
        Args:
            device: Target device (e.g. 'cpu', 'cuda:0')
            
        Returns:
            Memory bank instance with updated device
        """
        self._C = self._C.to(device)
        self._M = self._M.to(device)
        self._mean = self._mean.to(device)
        return self
    
class FasterDMD(BaseMemoryBank):
    """
    A compressed memory bank using matrix decomposition with interactive operations.
    
    Attributes:
        capacity: Current number of stored memory snippets (read-only)
        feature_dim: Original feature dimension (read-only)
        latent_dim: Compressed latent dimension (read-only)
    """
    
    def __init__(self, bank: torch.Tensor, latent_dim: int = 64):
        """
        Initialize memory bank with existing snippets.
        
        Args:
            bank: Initial memory snippets [N, feature_dim]
            latent_dim: Target latent dimension for compression
        """
        self._initialize_from_data(bank, latent_dim)

    @classmethod
    def from_state_dict(cls, path: str):
        """
        Load memory bank from saved state dictionary.
        
        Args:
            state_dict: Dictionary containing:
                - 'C': Coefficient matrix [N, d]
                - 'M': Basis matrix [feature_dim, d]
                - 'S': Singular values [d]
                - 'mean': Mean vector [feature_dim]
                - 'latent_dim': Latent dimension
                - 'feature_dim': Feature dimension
        """
        instance = cls.__new__(cls)
        state_dict = torch.load(path)
        assert type(state_dict) == dict
        instance._C = state_dict['C']
        instance._M = state_dict['M']
        instance._S = state_dict['S']
        instance._mean = state_dict['mean']
        instance._latent_dim = state_dict['latent_dim']
        instance._feature_dim = state_dict['feature_dim']
        instance.original_data = instance._C @ instance._M.T + instance._mean
        return instance

    def save(self, file_path: str):
        """
        Save current state to file.
        
        Args:
            file_path: Path to save the state dictionary
        """
        state_dict = {
            'C': self._C,
            'M': self._M,
            'S': self._S,
            'mean': self._mean,
            'latent_dim': self._latent_dim,
            'feature_dim': self._feature_dim
        }
        torch.save(state_dict, file_path)
        
    def feature_dim(self) -> int:
        """Original feature dimension of memory snippets"""
        return self._feature_dim

    def _initialize_from_data(self, bank: torch.Tensor, latent_dim: int):
        """Internal initialization from raw data"""
        # Center data and perform SVD
        self._mean = bank.mean(dim=0)
        centered = bank - self._mean
        U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
        
        # Truncate to latent dimension
        self._S = S[:latent_dim]
        self._C = (U[:, :latent_dim] * torch.sqrt(self._S.unsqueeze(0))).clone()
        self._M = (Vh.T[:, :latent_dim] * torch.sqrt(self._S.unsqueeze(0))).clone()
        
        self.original_data:torch.Tensor = bank.clone()
        
        # Public attributes
        self._latent_dim = latent_dim
        self._feature_dim = bank.shape[1]
    
    def _refresh(self):
        self.original_data = self._C @ self._M.T + self._mean

    @property
    def capacity(self) -> int:
        """Current number of stored memory snippets"""
        return self._C.shape[0]

    def get(self, index: int) -> torch.Tensor:
        """
        Retrieve a memory snippet by index.
        
        Args:
            index: Position in memory bank (0 <= index < capacity)
            
        Returns:
            Decompressed snippet tensor [feature_dim]
        """
        if self.original_data is None:
            self._refresh()
        self._validate_index(index)
        return self.original_data[index]

    def add_exact(self, snippet: torch.Tensor) -> torch.Tensor:
        """
        Add a new snippet through exact projection.
        
        Args:
            snippet: New feature vector [feature_dim]
            
        Returns:
            Added snippet in decompressed form [feature_dim]
        """
        # Project to latent space
        centered = snippet - self._mean
        coeff = (centered @ self._M) / self._S
        
        # Update storage
        self._C = torch.cat([self._C, coeff.unsqueeze(0)], dim=0)
        self._refresh()
        return self.get(-1)

    def add_interpolated(self, i: int, j: int, alpha: float = 0.5) -> torch.Tensor:
        """
        Generate new snippet by interpolating between existing entries.
        
        Args:
            i: Index of first base snippet
            j: Index of second base snippet
            alpha: Interpolation weight (0.0-1.0)
            
        Returns:
            New interpolated snippet [feature_dim]
        """
        self._validate_index(i)
        self._validate_index(j)
        
        # Linear interpolation in latent space
        interp_coeff = alpha * self._C[i] + (1 - alpha) * self._C[j]
        self._C = torch.cat([self._C, interp_coeff.unsqueeze(0)], dim=0)
        self._refresh()
        return self.get(-1)

    def _validate_index(self, index: int):
        """Ensure index is within valid range"""
        # if not index < self.capacity:
        #     raise IndexError(f"Index {index} out of range [0, {self.capacity-1}]")
        pass

    def __getitem__(self, index: int) -> torch.Tensor:
        """Enable direct indexing (memory_bank[i])"""
        return self.get(index)

    def __len__(self) -> int:
        """Return current memory capacity"""
        return self.capacity
    
    def to(self, device):
        """
        Move memory bank to a new device.
        
        Args:
            device: Target device (e.g. 'cpu', 'cuda:0')
            
        Returns:
            Memory bank instance with updated device
        """
        self._C = self._C.to(device)
        self._M = self._M.to(device)
        self._mean = self._mean.to(device)
        self._original_data = self._original_data.to(device)
        return self    
    

bank_models = {
    'dmd': DMDMemoryBank,
    'full': FullMemoryBank,
    'dmdfaster': FasterDMD
}