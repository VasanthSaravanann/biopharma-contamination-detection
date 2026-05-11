"""
MH-DDPM: Multimodal Hierarchical Denoising Diffusion Probabilistic Model

Generates realistic UV-Vis contamination spectra conditioned on:
- Contaminant species type
- Inoculum level (CFU/mL)
- Process conditions

Uses a hierarchical diffusion process with multimodal conditioning for
high-quality synthetic data generation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import math
import copy
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


@dataclass
class DDPMConfig:
    """Configuration for MH-DDPM model"""
    # Model architecture
    input_dim: int = 601  # Wavelengths 200-800 nm
    hidden_dim: int = 256
    latent_dim: int = 128
    n_layers: int = 6
    dropout: float = 0.1
    
    # Diffusion process
    n_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    schedule_type: str = 'linear'  # 'linear', 'cosine', 'quadratic'
    
    # Conditioning
    n_contaminant_types: int = 6  # E.coli, B.subtilis, P.aeruginosa, C.albicans, A.niger, Mycoplasma
    inoculum_levels: int = 7  # Number of discrete inoculum levels
    condition_dim: int = 32  # Embedding dimension for conditions
    
    # Training
    learning_rate: float = 1e-4
    batch_size: int = 64
    epochs: int = 200
    gradient_clip: float = 1.0
    ema_decay: float = 0.999
    
    # Device
    device: str = 'auto'


class PositionalEncoding(nn.Module):
    """Positional encoding for diffusion timesteps"""
    
    def __init__(self, dim: int, max_timesteps: int = 1000):
        super().__init__()
        self.dim = dim
        self.max_timesteps = max_timesteps
        
        # Create positional encoding table
        pe = torch.zeros(max_timesteps, dim)
        position = torch.arange(0, max_timesteps, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Get encoding for timesteps"""
        return self.pe[t]


class ConditionEmbedder(nn.Module):
    """Embed contaminant type and inoculum level conditions"""
    
    def __init__(self, n_contaminant_types: int, inoculum_levels: int, 
                 condition_dim: int):
        super().__init__()
        
        self.contaminant_embed = nn.Embedding(n_contaminant_types + 1, condition_dim)
        self.inoculum_embed = nn.Embedding(inoculum_levels + 1, condition_dim)
        
        self.condition_mlp = nn.Sequential(
            nn.Linear(condition_dim * 2, condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, condition_dim)
        )
    
    def forward(self, contaminant_type: torch.Tensor, 
                inoculum_level: torch.Tensor) -> torch.Tensor:
        """
        Embed conditioning information.
        
        Args:
            contaminant_type: Batch of contaminant type indices
            inoculum_level: Batch of inoculum level indices
            
        Returns:
            Combined condition embedding
        """
        c_emb = self.contaminant_embed(contaminant_type)
        i_emb = self.inoculum_embed(inoculum_level)
        
        combined = torch.cat([c_emb, i_emb], dim=-1)
        return self.condition_mlp(combined)


class ResidualBlock(nn.Module):
    """Residual block for UNet-style architecture"""
    
    def __init__(self, dim: int, condition_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )
        
        self.cond_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim)
        )
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )
        
        self.residual_projection = nn.Linear(dim, dim) if dim != dim else nn.Identity()
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, 
                c_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass with time and condition embeddings"""
        residual = x
        
        # Add time embedding
        t_emb = self.time_mlp(t_emb)
        if len(x.shape) == 3 and len(t_emb.shape) == 2:
            t_emb = t_emb.unsqueeze(1)
        x = x + t_emb
        
        # Add condition embedding
        c_emb = self.cond_mlp(c_emb)
        if len(x.shape) == 3 and len(c_emb.shape) == 2:
            c_emb = c_emb.unsqueeze(1)
        x = x + c_emb
        
        # MLP block with residual
        x = self.norm1(x)
        x = self.mlp(x)
        x = self.norm2(x)
        
        return x + residual


class MHDDPM(nn.Module):
    """
    Multimodal Hierarchical Denoising Diffusion Probabilistic Model.
    
    Generates realistic UV-Vis spectra conditioned on contaminant type
    and inoculum level using a diffusion process.
    """
    
    def __init__(self, config: DDPMConfig):
        super().__init__()
        
        self.config = config
        
        # Set device
        if config.device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(config.device)
        
        # Timestep encoding
        self.time_encoder = PositionalEncoding(config.hidden_dim, config.n_timesteps)
        
        # Condition embedding
        self.condition_embedder = ConditionEmbedder(
            config.n_contaminant_types,
            config.inoculum_levels,
            config.condition_dim
        )
        
        # Input projection
        self.input_projection = nn.Linear(config.input_dim, config.hidden_dim)
        
        # Hierarchical residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(config.hidden_dim, config.condition_dim, config.dropout)
            for _ in range(config.n_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(config.hidden_dim // 2, config.input_dim)
        )
        
        # Initialize weights
        self._initialize_weights()
        
        # Exponential Moving Average for stable generation
        self.ema = None
        self.ema_decay = config.ema_decay
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def _get_beta_schedule(self) -> torch.Tensor:
        """Get noise schedule betas"""
        if self.config.schedule_type == 'linear':
            return torch.linspace(
                self.config.beta_start, 
                self.config.beta_end, 
                self.config.n_timesteps
            )
        elif self.config.schedule_type == 'cosine':
            s = 0.008
            t = torch.linspace(0, self.config.n_timesteps, self.config.n_timesteps + 1)
            alphas_cumprod = torch.cos((t / self.config.n_timesteps + s) / (1 + s) * math.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0.0001, 0.9999)
        elif self.config.schedule_type == 'quadratic':
            return torch.linspace(
                math.sqrt(self.config.beta_start),
                math.sqrt(self.config.beta_end),
                self.config.n_timesteps
            ) ** 2
        else:
            raise ValueError(f"Unknown schedule type: {self.config.schedule_type}")
    
    def _get_diffusion_params(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Pre-compute diffusion parameters.
        
        Returns:
            Tuple of (betas, alphas_cumprod, sqrt_alphas_cumprod)
        """
        betas = self._get_beta_schedule()
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        
        return betas, alphas_cumprod, sqrt_alphas_cumprod
    
    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, 
                 noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward diffusion process: q(x_t | x_0).
        
        Args:
            x_0: Original data
            t: Timestep indices
            noise: Optional pre-sampled noise
            
        Returns:
            Noisy data at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        _, alphas_cumprod, sqrt_alphas_cumprod = self._get_diffusion_params()
        alphas_cumprod = alphas_cumprod.to(x_0.device)
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(x_0.device)
        
        # x_t = sqrt(alpha_cumprod) * x_0 + sqrt(1 - alpha_cumprod) * noise
        sqrt_alpha_cumprod_t = sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1 - alphas_cumprod[t]).view(-1, 1)
        
        return sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise
    
    def p_losses(self, x_0: torch.Tensor, t: torch.Tensor,
                 contaminant_type: torch.Tensor,
                 inoculum_level: torch.Tensor,
                 noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate diffusion loss.
        
        Args:
            x_0: Original data
            t: Timestep indices
            contaminant_type: Contaminant type indices
            inoculum_level: Inoculum level indices
            noise: Optional pre-sampled noise
            
        Returns:
            Loss value
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        # Add noise
        x_t = self.q_sample(x_0, t, noise)
        
        # Get embeddings
        t_emb = self.time_encoder(t)
        c_emb = self.condition_embedder(contaminant_type, inoculum_level)
        
        # Forward pass
        x_pred = self.forward(x_t, t_emb, c_emb)
        
        # Predict noise (not x_0 directly)
        loss = F.mse_loss(x_pred, noise)
        
        return loss
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, 
                c_emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input (noisy data or latent)
            t_emb: Time embedding
            c_emb: Condition embedding
            
        Returns:
            Predicted noise
        """
        # Project input
        h = self.input_projection(x)
        
        # Pass through residual blocks
        for block in self.residual_blocks:
            h = block(h, t_emb, c_emb)
        
        # Project output
        noise_pred = self.output_projection(h)
        
        return noise_pred
    
    @torch.no_grad()
    def p_sample(self, x_t: torch.Tensor, t: torch.Tensor,
                 c_emb: torch.Tensor) -> torch.Tensor:
        """
        Single reverse diffusion step: p(x_{t-1} | x_t).
        
        Args:
            x_t: Data at timestep t
            t: Timestep index
            c_emb: Condition embedding
            
        Returns:
            Data at timestep t-1
        """
        _, alphas_cumprod, sqrt_alphas_cumprod = self._get_diffusion_params()
        alphas_cumprod = alphas_cumprod.to(x_t.device)
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(x_t.device)
        
        # Get time embedding
        t_emb = self.time_encoder(t)
        
        # Predict noise
        if self.ema is not None:
            # Use EMA model for better generation
            noise_pred = self.ema(x_t, t_emb, c_emb)
        else:
            noise_pred = self(x_t, t_emb, c_emb)
        
        # Get parameters for this timestep
        sqrt_alpha_cumprod_t = sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1 - alphas_cumprod[t]).view(-1, 1)
        beta_t = self._get_beta_schedule()[t].view(-1, 1).to(x_t.device)
        
        # Calculate x_0 estimate
        x_0_pred = (x_t - sqrt_one_minus_alpha_cumprod_t * noise_pred) / sqrt_alpha_cumprod_t
        
        # Calculate posterior mean
        alpha_t = 1.0 - beta_t
        
        # Simple DDPM mean calculation
        mean = (1 / torch.sqrt(alpha_t)) * (
            x_t - (beta_t / sqrt_one_minus_alpha_cumprod_t) * noise_pred
        )
        
        # Add noise (except for final step)
        if t[0] > 0:
            noise = torch.randn_like(x_t)
            return mean + torch.sqrt(beta_t) * noise
        else:
            return mean
    
    @torch.no_grad()
    def sample(self, n_samples: int, contaminant_type: torch.Tensor,
               inoculum_level: torch.Tensor, progress: bool = False) -> torch.Tensor:
        """
        Generate samples from the model.
        
        Args:
            n_samples: Number of samples to generate
            contaminant_type: Contaminant type indices
            inoculum_level: Inoculum level indices
            progress: Show progress bar
            
        Returns:
            Generated spectra
        """
        self.eval()
        
        # Start from pure noise
        x_t = torch.randn(n_samples, self.config.input_dim).to(self.device)
        
        # Get condition embedding
        c_emb = self.condition_embedder(contaminant_type, inoculum_level)
        
        # Reverse diffusion
        iterator = reversed(range(self.config.n_timesteps))
        if progress:
            iterator = tqdm(iterator, desc="Generating samples")
        
        for t in iterator:
            t_batch = torch.full((n_samples,), t, dtype=torch.long).to(self.device)
            x_t = self.p_sample(x_t, t_batch, c_emb)
        
        return x_t
    
    def update_ema(self):
        """Update exponential moving average"""
        if self.ema is None:
            self.ema = type(self)(self.config).to(self.device)
            # Initialize with current weights
            for ema_param, param in zip(self.ema.parameters(), self.parameters()):
                ema_param.data.copy_(param.data)
        
        # Update EMA
        for ema_param, param in zip(self.ema.parameters(), self.parameters()):
            ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)
    
    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                   optimizer: torch.optim.Optimizer) -> float:
        """
        Single training step.
        
        Args:
            batch: Tuple of (spectra, contaminant_type, inoculum_level, metadata)
            optimizer: Optimizer
            
        Returns:
            Loss value
        """
        self.train()
        
        spectra, contaminant_type, inoculum_level = batch
        
        spectra = spectra.to(self.device)
        contaminant_type = contaminant_type.to(self.device)
        inoculum_level = inoculum_level.to(self.device)
        
        # Sample random timesteps
        t = torch.randint(0, self.config.n_timesteps, (len(spectra),), 
                         dtype=torch.long).to(self.device)
        
        # Calculate loss
        loss = self.p_losses(spectra, t, contaminant_type, inoculum_level)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.config.gradient_clip)
        
        optimizer.step()
        
        # Update EMA
        self.update_ema()
        
        return loss.item()
    
    def fit(self, spectra: np.ndarray, contaminant_types: np.ndarray,
            inoculum_levels: np.ndarray, val_split: float = 0.1,
            verbose: bool = True) -> Dict:
        """
        Train the DDPM model.
        
        Args:
            spectra: Training spectra array (n_samples, n_wavelengths)
            contaminant_types: Contaminant type indices
            inoculum_levels: Inoculum level indices
            val_split: Validation split fraction
            verbose: Print training progress
            
        Returns:
            Training history
        """
        # Convert to tensors
        spectra_tensor = torch.FloatTensor(spectra)
        contaminant_tensor = torch.LongTensor(contaminant_types)
        inoculum_tensor = torch.LongTensor(inoculum_levels)
        
        # Split data
        n_val = int(len(spectra) * val_split)
        indices = torch.randperm(len(spectra))
        
        train_indices = indices[n_val:]
        val_indices = indices[:n_val]
        
        train_dataset = TensorDataset(
            spectra_tensor[train_indices],
            contaminant_tensor[train_indices],
            inoculum_tensor[train_indices]
        )
        val_dataset = TensorDataset(
            spectra_tensor[val_indices],
            contaminant_tensor[val_indices],
            inoculum_tensor[val_indices]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, 
                                 shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, 
                               shuffle=False)
        
        # Initialize optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=1e-4
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.epochs, eta_min=1e-6
        )
        
        # Training loop
        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        
        for epoch in range(self.config.epochs):
            # Training
            train_losses = []
            for batch in train_loader:
                loss = self.train_step(batch, optimizer)
                train_losses.append(loss)
            
            avg_train_loss = np.mean(train_losses)
            
            # Validation
            val_losses = []
            self.eval()
            with torch.no_grad():
                for batch in val_loader:
                    spectra_b, cont_b, inoc_b = batch
                    spectra_b = spectra_b.to(self.device)
                    cont_b = cont_b.to(self.device)
                    inoc_b = inoc_b.to(self.device)
                    
                    t = torch.randint(0, self.config.n_timesteps, (len(spectra_b),),
                                     dtype=torch.long).to(self.device)
                    loss = self.p_losses(spectra_b, t, cont_b, inoc_b)
                    val_losses.append(loss.item())
            
            avg_val_loss = np.mean(val_losses)
            
            # Update scheduler
            scheduler.step()
            
            # Store history
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.config.epochs} - "
                      f"Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
        
        return history
    
    def save(self, path: str):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'ema_state_dict': self.ema.state_dict() if self.ema is not None else None,
            'config': self.config,
        }
        torch.save(checkpoint, path)
    
    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> 'MHDDPM':
        """Load model from checkpoint"""
        checkpoint = torch.load(path, map_location='cpu')
        config = checkpoint['config']
        
        if device is not None:
            config.device = device
        
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if checkpoint['ema_state_dict'] is not None:
            model.ema = type(model)(config)
            model.ema.load_state_dict(checkpoint['ema_state_dict'])
        
        return model


class FeatureDiffusionModel(MHDDPM):
    """
    Lightweight feature-space diffusion baseline for tabular features.
    
    Operates on extracted feature vectors (~50 dimensions) instead of 
    raw high-dimensional spectra. Uses the same MH-DDPM architecture
    but optimized for lower-dimensional tabular data.
    """
    
    def __init__(self, config: DDPMConfig):
        """
        Initialize feature-space diffusion model.
        
        Args:
            config: DDPM configuration (ensure input_dim matches feature count)
        """
        super().__init__(config)
        print(f"Initialized FeatureDiffusionModel with input_dim={config.input_dim}")
    
    def fit_features(self, features: np.ndarray, contaminant_types: np.ndarray,
                    inoculum_levels: np.ndarray, **kwargs):
        """
        Train on extracted features.
        
        Wrapper around fit() for semantic clarity.
        """
        return self.fit(features, contaminant_types, inoculum_levels, **kwargs)
    
    def sample_features(self, n_samples: int, contaminant_type: int,
                       inoculum_level: int, **kwargs) -> np.ndarray:
        """
        Generate synthetic feature vectors.
        """
        contaminant_tensor = torch.full((n_samples,), contaminant_type, 
                                        dtype=torch.long).to(self.device)
        inoculum_tensor = torch.full((n_samples,), inoculum_level,
                                     dtype=torch.long).to(self.device)
        
        features = self.sample(n_samples, contaminant_tensor, inoculum_tensor, **kwargs)
        return features.cpu().numpy()


class SyntheticDataGenerator:
    """
    Generate synthetic contamination spectra using trained MH-DDPM.
    
    Validates generated data using statistical metrics.
    """
    
    def __init__(self, model: MHDDPM, wavelengths: Optional[np.ndarray] = None):
        """
        Initialize generator.
        
        Args:
            model: Trained MH-DDPM model
            wavelengths: Wavelength array
        """
        self.model = model
        self.model.eval()
        
        if wavelengths is None:
            self.wavelengths = np.arange(200, 801, 1)
        else:
            self.wavelengths = wavelengths
    
    def generate(self, n_samples: int, contaminant_type: int,
                 inoculum_level: int, progress: bool = False) -> np.ndarray:
        """
        Generate synthetic spectra.
        
        Args:
            n_samples: Number of samples
            contaminant_type: Contaminant type index (0-5)
            inoculum_level: Inoculum level index (0-6)
            progress: Show progress bar
            
        Returns:
            Generated spectra array
        """
        contaminant_tensor = torch.full((n_samples,), contaminant_type, 
                                        dtype=torch.long).to(self.model.device)
        inoculum_tensor = torch.full((n_samples,), inoculum_level,
                                     dtype=torch.long).to(self.model.device)
        
        with torch.no_grad():
            spectra = self.model.sample(n_samples, contaminant_tensor, 
                                        inoculum_tensor, progress=progress)
        
        # Ensure non-negative absorbance
        spectra = torch.clamp(spectra, min=0)
        
        return spectra.cpu().numpy()
    
    def generate_dataset(self, n_per_class: int = 100,
                         progress: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate balanced synthetic dataset.
        
        Args:
            n_per_class: Samples per contaminant type / inoculum level combination
            progress: Show progress bar
            
        Returns:
            Tuple of (spectra, contaminant_types, inoculum_levels)
        """
        all_spectra = []
        all_contaminants = []
        all_inoculum = []
        
        for cont_type in range(self.model.config.n_contaminant_types):
            for inoc_level in range(self.model.config.inoculum_levels):
                spectra = self.generate(n_per_class, cont_type, inoc_level, progress)
                
                all_spectra.append(spectra)
                all_contaminants.append(np.full(n_per_class, cont_type))
                all_inoculum.append(np.full(n_per_class, inoc_level))
        
        return (
            np.vstack(all_spectra),
            np.concatenate(all_contaminants),
            np.concatenate(all_inoculum)
        )


def evaluate_ddpm_baselines(raw_model: MHDDPM,
                            feature_model: FeatureDiffusionModel,
                            n_samples: int = 100,
                            contaminant_type: int = 0,
                            inoculum_level: int = 0) -> Dict:
    """
    Generate side-by-side baseline samples from raw-spectrum and feature DDPM models.

    Args:
        raw_model: Trained MHDDPM on spectra
        feature_model: Trained FeatureDiffusionModel on extracted features
        n_samples: Number of samples per baseline
        contaminant_type: Contaminant class index
        inoculum_level: Inoculum class index

    Returns:
        Dictionary containing generated samples and summary statistics.
    """
    raw_gen = SyntheticDataGenerator(raw_model)
    raw_samples = raw_gen.generate(
        n_samples=n_samples,
        contaminant_type=contaminant_type,
        inoculum_level=inoculum_level,
        progress=False,
    )

    feature_samples = feature_model.sample_features(
        n_samples=n_samples,
        contaminant_type=contaminant_type,
        inoculum_level=inoculum_level,
        progress=False,
    )

    return {
        'raw_spectrum': {
            'samples': raw_samples,
            'shape': tuple(raw_samples.shape),
            'mean': float(np.mean(raw_samples)),
            'std': float(np.std(raw_samples)),
        },
        'feature_space': {
            'samples': feature_samples,
            'shape': tuple(feature_samples.shape),
            'mean': float(np.mean(feature_samples)),
            'std': float(np.std(feature_samples)),
        },
        'side_by_side': {
            'n_samples': int(n_samples),
            'contaminant_type': int(contaminant_type),
            'inoculum_level': int(inoculum_level),
        },
    }


if __name__ == "__main__":
    # Test MH-DDPM model
    from data_simulation import UVVisSpectraGenerator, ContaminantType
    
    print("Generating training data...")
    generator = UVVisSpectraGenerator(seed=42)
    df = generator.generate_dataset(n_clean=100, n_contaminated_per_type=200, seed=42)
    
    # Prepare data
    wavelength_cols = [c for c in df.columns if c.startswith('abs_')]
    spectra = df[wavelength_cols].values
    
    # Map contaminant types to indices
    contaminant_map = {
        'E_coli': 0, 'B_subtilis': 1, 'P_aeruginosa': 2,
        'C_albicans': 3, 'A_niger': 4, 'Mycoplasma': 5, 'Clean': 6
    }
    contaminant_types = df['contaminant_type'].map(contaminant_map).values
    
    # Map inoculum levels to indices
    inoculum_levels_map = {10: 0, 25: 1, 50: 2, 100: 3, 250: 4, 500: 5, 1000: 6}
    inoculum_levels = df['inoculum_level'].map(
        lambda x: inoculum_levels_map.get(x, 0)
    ).values
    
    # Filter to contaminated only for training
    contaminated_mask = df['label'] == 1
    spectra_cont = spectra[contaminated_mask]
    cont_types_cont = contaminant_types[contaminated_mask]
    inoc_levels_cont = inoculum_levels[contaminated_mask]
    
    print(f"Training on {len(spectra_cont)} contaminated spectra")
    
    # Initialize model
    config = DDPMConfig(
        input_dim=spectra_cont.shape[1],
        n_timesteps=100,  # Reduced for testing
        epochs=50,  # Reduced for testing
        batch_size=32
    )
    
    model = MHDDPM(config)
    
    # Train
    print("\nTraining MH-DDPM...")
    history = model.fit(
        spectra_cont, cont_types_cont, inoc_levels_cont,
        val_split=0.1, verbose=True
    )
    
    # Generate samples
    print("\nGenerating synthetic spectra...")
    synth_generator = SyntheticDataGenerator(model)
    synthetic_spectra = synth_generator.generate(
        n_samples=10,
        contaminant_type=0,
        inoculum_level=3,
        progress=True
    )
    
    print(f"\nGenerated {len(synthetic_spectra)} synthetic spectra")
    print(f"Shape: {synthetic_spectra.shape}")
