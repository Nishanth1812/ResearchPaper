"""
DigitalOcean GPU Droplet Training Script for DBGDGM Model
Optimized for RTX 6000 Ada GPU

Usage:
    python do_train.py --dataset hcp --categorical-dim 8 --trial 1 --data-dir ../data
    python do_train.py --dataset hcp --categorical-dim 8 --trial 1 --data-dir ../data --fast
"""
import argparse
import logging
import sys
import time
from pathlib import Path

# Add parent directory to path for src imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch


def setup_logging(save_dir: Path, dataset: str, trial: int):
    """Setup logging to both file and console."""
    log_file = save_dir / f"fMRI_{dataset}_{trial}.log"
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(levelname)s:%(message)s'))
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def setup_gpu():
    """Setup GPU with optimizations for RTX 6000 Ada."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 60)
    print("🧠 DBGDGM Training on DigitalOcean GPU Droplet")
    print("=" * 60)
    print(f"🚀 Device: {device}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"🎮 GPU: {gpu_name}")
        print(f"💾 GPU Memory: {gpu_memory:.1f} GB")
        
        # Enable optimizations for Ada Lovelace architecture
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        print("⚡ TF32 and cuDNN benchmark enabled for RTX 6000 Ada")
    else:
        print("⚠️  No GPU detected, using CPU (this will be slow!)")
    
    return device


def train_optimized(model, dataset, save_path, num_epochs, batch_size, learning_rate,
                    device, temp_min, anneal_rate, valid_prop, test_prop, temp,
                    eval_every=50, patience=100):
    """
    Optimized training loop with:
    - Less frequent validation (every eval_every epochs)
    - Early stopping with patience
    - Progress tracking
    """
    from src.dataset import data_loader
    
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.to(device)
    
    best_nll = float('inf')
    best_nll_train = float('inf')
    epochs_without_improvement = 0
    start_time = time.time()
    
    print(f"\n📊 Training Configuration:")
    print(f"   Epochs: {num_epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Eval every: {eval_every} epochs")
    print(f"   Early stopping patience: {patience} epochs")
    print(f"   Learning rate: {learning_rate}")
    print()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        np.random.shuffle(dataset)
        model.train()
        
        running_loss = {'nll': 0, 'kld_z': 0, 'kld_alpha': 0, 'kld_beta': 0, 'kld_phi': 0}
        
        for batch_graphs in data_loader(dataset, batch_size):
            optimizer.zero_grad()
            
            batch_loss = model(batch_graphs, valid_prop=valid_prop, test_prop=test_prop, temp=temp)
            
            loss = (batch_loss['nll'] + batch_loss['kld_z']) / len(batch_graphs)
            loss.backward()
            optimizer.step()
            
            for loss_name in running_loss.keys():
                running_loss[loss_name] += batch_loss[loss_name].cpu().detach().data.numpy() / len(dataset)
        
        epoch_time = time.time() - epoch_start
        elapsed = time.time() - start_time
        
        # Progress logging
        if epoch % 10 == 0:
            eta = (elapsed / (epoch + 1)) * (num_epochs - epoch - 1)
            print(f"Epoch {epoch:4d}/{num_epochs} | NLL: {running_loss['nll']:.4f} | "
                  f"Time: {epoch_time:.1f}s | Elapsed: {elapsed/60:.1f}m | ETA: {eta/60:.1f}m")
        
        # Validation (only every eval_every epochs)
        if epoch % eval_every == 0 or epoch == num_epochs - 1:
            model.eval()
            with torch.no_grad():
                nll, aucroc, ap = model.predict_auc_roc_precision(
                    dataset, valid_prop=valid_prop, test_prop=test_prop)
            
            print(f"\n{'='*60}")
            print(f"📈 Validation at Epoch {epoch}")
            print(f"   Train  - NLL: {nll['train']:.4f}, AUC: {aucroc['train']:.4f}, AP: {ap['train']:.4f}")
            print(f"   Valid  - NLL: {nll['valid']:.4f}, AUC: {aucroc['valid']:.4f}, AP: {ap['valid']:.4f}")
            print(f"   Test   - NLL: {nll['test']:.4f}, AUC: {aucroc['test']:.4f}, AP: {ap['test']:.4f}")
            print(f"{'='*60}\n")
            
            logging.info(f"Epoch {epoch} | train nll {nll['train']} aucroc {aucroc['train']} ap {ap['train']} | "
                        f"valid nll {nll['valid']} aucroc {aucroc['valid']} ap {ap['valid']} | "
                        f"test nll {nll['test']} aucroc {aucroc['test']} ap {ap['test']}")
            
            # Save best validation model
            if nll['valid'] < best_nll:
                print(f"💾 New best validation NLL: {nll['valid']:.4f} (improved by {best_nll - nll['valid']:.4f})")
                embeddings = model.predict_embeddings(dataset, valid_prop=valid_prop, test_prop=test_prop)
                torch.save((model.state_dict(), optimizer.state_dict()), save_path / "checkpoint.pt")
                np.save(save_path / "results.npy", {
                    'nll': nll, 'aucroc': aucroc, 'ap': ap, 'embeddings': embeddings
                })
                best_nll = nll['valid']
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += eval_every
            
            # Save best train model
            if nll['train'] < best_nll_train:
                embeddings = model.predict_embeddings(dataset, valid_prop=valid_prop, test_prop=test_prop)
                torch.save((model.state_dict(), optimizer.state_dict()), save_path / "checkpoint_best_train.pt")
                np.save(save_path / "results_best_train.npy", {
                    'nll': nll, 'aucroc': aucroc, 'ap': ap, 'embeddings': embeddings
                })
                best_nll_train = nll['train']
            
            # Early stopping
            if epochs_without_improvement >= patience:
                print(f"\n🛑 Early stopping triggered after {epoch} epochs (no improvement for {patience} epochs)")
                break
        
        # Temperature annealing
        if epoch % 10 == 0:
            temp = np.maximum(temp * np.exp(-anneal_rate * epoch), temp_min)
            learning_rate *= 0.99
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
    
    total_time = time.time() - start_time
    print(f"\n✅ Training complete in {total_time/3600:.2f} hours")
    print(f"   Best validation NLL: {best_nll:.4f}")
    print(f"   Best train NLL: {best_nll_train:.4f}")
    print(f"   Model saved to: {save_path}")


def main(args):
    from src.dataset import load_dataset
    from src.model import Model
    from src.train import train
    
    # Setup save directory (use output-dir if specified, otherwise local)
    if args.output_dir:
        save_dir = Path(args.output_dir) / f"models_{args.dataset}_{args.trial}"
    else:
        save_dir = Path.cwd() / f"models_{args.dataset}_{args.trial}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    setup_logging(save_dir, args.dataset, args.trial)
    
    # Setup GPU
    device = setup_gpu()
    
    # Hyperparameters
    dataset_args = dict(
        dataset=args.dataset,
        window_size=30,
        window_stride=30,
        measure="correlation",
        top_percent=5
    )
    
    model_args = dict(
        sigma=1.0,
        gamma=0.1,
        categorical_dim=args.categorical_dim,
        embedding_dim=128
    )
    
    train_args = dict(
        num_epochs=args.num_epochs,
        save_path=save_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=device,
        temp_min=0.05,
        anneal_rate=3e-4,
        valid_prop=args.valid_prop,
        test_prop=args.test_prop,
        temp=1.0
    )
    
    # Log configuration
    logging.debug('Dataset args: %s', dataset_args)
    logging.debug('Model args: %s', model_args)
    logging.debug('Train args: %s', train_args)
    
    # Load dataset
    # Convert to absolute path to avoid issues with relative paths
    data_dir = Path(args.data_dir).resolve()
    
    # Validate data directory exists
    raw_data_dir = data_dir / args.dataset / "raw"
    print(f"\n📊 Loading {args.dataset.upper()} dataset...")
    print(f"   Data directory: {data_dir}")
    print(f"   Raw data path: {raw_data_dir}")
    
    if not data_dir.exists():
        print(f"\n❌ ERROR: Data directory does not exist: {data_dir}")
        print(f"\n   Please extract the dataset first:")
        print(f"   cd {data_dir.parent}")
        print(f"   unzip dataset.zip")
        sys.exit(1)
    
    if not raw_data_dir.exists():
        print(f"\n❌ ERROR: Raw data directory does not exist: {raw_data_dir}")
        print(f"\n   Expected structure:")
        print(f"   {data_dir}/")
        print(f"   └── {args.dataset}/")
        print(f"       └── raw/")
        print(f"           └── *.npy files")
        print(f"\n   Please check your data extraction:")
        print(f"   ls -la {data_dir}")
        print(f"   ls -la {data_dir}/{args.dataset}/")
        sys.exit(1)
    
    # Check for .npy files
    npy_files = list(raw_data_dir.glob("*.npy"))
    if len(npy_files) == 0:
        print(f"\n❌ ERROR: No .npy files found in {raw_data_dir}")
        print(f"\n   Please extract the dataset:")
        print(f"   cd {data_dir.parent}")
        print(f"   unzip -o dataset.zip")
        sys.exit(1)
    
    print(f"   Found {len(npy_files)} .npy files")
    
    dataset = load_dataset(**dataset_args, data_dir=str(data_dir))
    
    if len(dataset) == 0:
        print(f"\n❌ ERROR: Dataset is empty after loading!")
        print(f"   This might mean the .npy files are corrupted or in wrong format.")
        sys.exit(1)
    
    num_subjects, num_nodes = len(dataset), dataset[0][1][0].number_of_nodes()
    print(f"✅ Loaded {num_subjects} subjects with {num_nodes} nodes each")
    
    # Initialize model
    print(f"\n🏗️  Initializing model...")
    model = Model(num_subjects, num_nodes, **model_args, device=device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Train
    print(f"\n🏋️  Starting training...")
    
    if args.fast:
        # Optimized training with less frequent validation and early stopping
        train_optimized(
            model=model,
            dataset=dataset,
            save_path=save_dir,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=device,
            temp_min=0.05,
            anneal_rate=3e-4,
            valid_prop=args.valid_prop,
            test_prop=args.test_prop,
            temp=1.0,
            eval_every=args.eval_every,
            patience=args.patience
        )
    else:
        # Original training loop
        train(model, dataset, **train_args)
    
    print(f"\n🎉 Training finished! Results saved to: {save_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='DBGDGM model training on DigitalOcean GPU Droplet',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--dataset', required=True, type=str, choices=['ukb', 'hcp'],
                        help='Dataset to train on')
    parser.add_argument('--categorical-dim', required=True, type=int,
                        help='Number of brain communities')
    parser.add_argument('--trial', required=True, type=int,
                        help='Trial number for experiment tracking')
    
    # Training parameters
    parser.add_argument('--num-epochs', default=1001, type=int,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', default=1, type=int,
                        help='Training batch size')
    parser.add_argument('--learning-rate', default=1e-4, type=float,
                        help='Initial learning rate')
    parser.add_argument('--valid-prop', default=0.1, type=float,
                        help='Proportion of data for validation')
    parser.add_argument('--test-prop', default=0.1, type=float,
                        help='Proportion of data for testing')
    
    # Data path
    parser.add_argument('--data-dir', default='./data', type=str,
                        help='Directory containing the dataset')
    
    # Optimization flags
    parser.add_argument('--fast', action='store_true',
                        help='Enable optimized training (less frequent validation, early stopping)')
    parser.add_argument('--eval-every', default=50, type=int,
                        help='Evaluate every N epochs (only with --fast)')
    parser.add_argument('--patience', default=150, type=int,
                        help='Early stopping patience in epochs (only with --fast)')
    
    # Output directory (for saving to external storage volume)
    parser.add_argument('--output-dir', default=None, type=str,
                        help='Directory to save models/checkpoints (e.g., /mnt/storage/dbgdgm_models)')
    
    args = parser.parse_args()
    main(args)
