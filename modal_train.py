"""
Modal Cloud GPU Training Script for DBGDGM Model
Optimized for A100-40GB GPU

Run: modal run modal_train.py --dataset ukb --categorical-dim 8 --trial 1
"""
import modal

app = modal.App("dbgdgm-training")

# Persistent volumes
data_volume = modal.Volume.from_name("dbgdgm-data", create_if_missing=True)
models_volume = modal.Volume.from_name("dbgdgm-models", create_if_missing=True)

# Container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        # Core ML - use CUDA-optimized torch
        "torch",
        "numpy==1.23.4",
        "scipy==1.9.3",
        "scikit-learn==1.1.3",
        # Graph & Brain imaging
        "networkx==2.8.8",
        "nibabel==4.0.2",
        "nilearn==0.9.2",
        # Utilities
        "pandas==1.5.1",
        "matplotlib==3.6.2",
        "tqdm==4.64.1",
        "lxml==4.9.1",
    )
    .add_local_python_source("src")  # Include your src/ module
)


@app.function(
    image=image,
    gpu="L40S",  # A100-40GB for fast training
    timeout=86400,  # 24 hours max
    volumes={
        "/vol": data_volume,
        "/models": models_volume,
    },
)
def train_model(
    dataset: str = "ukb",
    categorical_dim: int = 8,
    valid_prop: float = 0.1,
    test_prop: float = 0.1,
    trial: int = 1,
    num_epochs: int = 1001,
):
    """Train DBGDGM model on Modal A100 GPU."""
    import logging
    import sys
    from pathlib import Path
    import torch

    from src.dataset import load_dataset
    from src.model import Model
    from src.train import train

    # Setup save directory
    save_dir = Path(f"/models/models_{dataset}_{trial}")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Enhanced logging - both file AND console output
    log_file = save_dir / f"fMRI_{dataset}_{trial}.log"
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # File handler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(levelname)s:%(message)s'))
    
    # Console handler - shows in Modal terminal
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    print("=" * 60)
    print("🧠 DBGDGM Training on Modal A100")
    print("=" * 60)

    # GPU setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        # Enable TF32 for faster training on A100
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("⚡ TF32 enabled for faster A100 training")

    # Load dataset - note: data is at /vol/data/ due to upload structure
    print("📊 Loading dataset...")
    dataset_loaded = load_dataset(
        dataset=dataset,
        window_size=30,
        window_stride=30,
        measure="correlation",
        top_percent=5,
        data_dir="/vol/data",  # Your data is at /vol/data/
    )
    num_subjects = len(dataset_loaded)
    num_nodes = dataset_loaded[0][1][0].number_of_nodes()
    print(f"✅ Loaded {num_subjects} subjects, {num_nodes} nodes each")

    # Initialize model
    model = Model(
        num_samples=num_subjects,
        num_nodes=num_nodes,
        sigma=1.0,
        gamma=0.1,
        categorical_dim=categorical_dim,
        embedding_dim=128,
        device=device,
    )

    # Train
    print(f"🏋️ Training for {num_epochs} epochs...")
    train(
        model=model,
        dataset=dataset_loaded,
        save_path=save_dir,
        num_epochs=num_epochs,
        batch_size=1,
        learning_rate=1e-4,
        device=device,
        temp_min=0.05,
        anneal_rate=3e-4,
        valid_prop=valid_prop,
        test_prop=test_prop,
        temp=1.0,
    )

    # Commit changes to volume
    models_volume.commit()
    print(f"✅ Training complete! Model saved to {save_dir}")

    return str(save_dir)


@app.local_entrypoint()
def main(
    dataset: str = "ukb",
    categorical_dim: int = 8,
    valid_prop: float = 0.1,
    test_prop: float = 0.1,
    trial: int = 1,
    num_epochs: int = 1001,
):
    """Launch training on Modal cloud."""
    print("🚀 Launching DBGDGM training on Modal A100...")
    print(f"   Dataset: {dataset}")
    print(f"   Categorical dim: {categorical_dim}")
    print(f"   Epochs: {num_epochs}")
    print(f"   Trial: {trial}")

    result = train_model.remote(
        dataset=dataset,
        categorical_dim=categorical_dim,
        valid_prop=valid_prop,
        test_prop=test_prop,
        trial=trial,
        num_epochs=num_epochs,
    )
    print(f"✅ {result}")
