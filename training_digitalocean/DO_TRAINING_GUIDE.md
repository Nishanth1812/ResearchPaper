# DigitalOcean GPU Droplet Training Guide

## Quick Start

### 1. Create GPU Droplet

- Go to DigitalOcean → Create → Droplets
- Choose **GPU Droplet** with RTX 6000 Ada
- Select **Ubuntu 22.04** image
- **Attach your storage volume** (e.g., `/mnt/storage`)
- Add your SSH key

### 2. Connect & Setup

```bash
# SSH into your droplet
ssh root@<your-droplet-ip>

# Clone repository
git clone <your-repo-url>
cd ResearchPaper

# Run setup script (from training_digitalocean folder)
cd training_digitalocean
chmod +x setup_do_droplet.sh
./setup_do_droplet.sh
```

The setup script will:

- Install uv (fast Python package manager) with pip fallback
- Create virtual environment at `$REPO_DIR/.venv`
- Install PyTorch with CUDA 12.1 for RTX 6000 Ada
- Extract `dataset.zip` automatically
- Setup storage volume directory

### 3. Run Training

```bash
# Activate environment
source ~/ResearchPaper/.venv/bin/activate
cd ~/ResearchPaper/training_digitalocean

# Standard training (~4-5 hours) - saves to storage volume
python do_train.py --dataset hcp --categorical-dim 8 --trial 1 \
    --data-dir ../data --output-dir /mnt/storage/dbgdgm_models

# Fast training with optimizations (~1.5-2 hours)
python do_train.py --dataset hcp --categorical-dim 8 --trial 1 \
    --data-dir ../data --output-dir /mnt/storage/dbgdgm_models --fast

# Run in background (survives SSH disconnect)
nohup python do_train.py --dataset hcp --categorical-dim 8 --trial 1 \
    --data-dir ../data --output-dir /mnt/storage/dbgdgm_models --fast > training.log 2>&1 &
tail -f training.log  # Watch progress
```

---

## Training Options

| Flag                | Description                             | Default     |
| ------------------- | --------------------------------------- | ----------- |
| `--dataset`         | Dataset: `ukb` or `hcp`                 | Required    |
| `--categorical-dim` | Number of communities                   | Required    |
| `--trial`           | Experiment trial number                 | Required    |
| `--num-epochs`      | Training epochs                         | 1001        |
| `--batch-size`      | Batch size                              | 1           |
| `--learning-rate`   | Initial LR                              | 1e-4        |
| `--data-dir`        | Dataset directory                       | `./data`    |
| `--output-dir`      | Where to save models/checkpoints        | Current dir |
| `--fast`            | Enable optimizations                    | False       |
| `--eval-every`      | Validation frequency (with `--fast`)    | 50          |
| `--patience`        | Early stopping patience (with `--fast`) | 150         |

---

## Storage Volume

Models and checkpoints are saved to your attached storage volume:

```
/mnt/storage/dbgdgm_models/
└── models_hcp_1/
    ├── checkpoint.pt           # Best validation model
    ├── checkpoint_best_train.pt # Best training model
    ├── results.npy             # Metrics & embeddings
    ├── results_best_train.npy
    └── fMRI_hcp_1.log          # Training logs
```

---

## Expected Performance (RTX 6000 Ada)

| Mode     | Est. Time   | Notes                                       |
| -------- | ----------- | ------------------------------------------- |
| Standard | 4-5 hours   | Full validation every epoch                 |
| Fast     | 1.5-2 hours | Validation every 50 epochs + early stopping |

---

## Monitoring

```bash
# GPU usage
nvtop

# Training logs
tail -f /mnt/storage/dbgdgm_models/models_hcp_1/fMRI_hcp_1.log

# Or if running with nohup
tail -f training.log

# Check if training is running
ps aux | grep python
```

---

## Download Results

```bash
# From local machine
scp -r root@<droplet-ip>:/mnt/storage/dbgdgm_models/models_hcp_1 ./results/
```

---

## Troubleshooting

### Dataset not found

```bash
# Check dataset was extracted
ls -la ~/ResearchPaper/data/hcp/

# Manually extract if needed
cd ~/ResearchPaper
unzip dataset.zip
```

### Storage volume not mounted

```bash
# Check mounts
df -h

# Mount manually (replace with your volume name)
sudo mount -o defaults /dev/disk/by-id/<volume-id> /mnt/storage
```

### uv not found after setup

```bash
# Add to PATH
export PATH="$HOME/.cargo/bin:$PATH"
# Or use pip instead
pip install <package>
```

### src module not found

```bash
# Make sure you're running from the right directory
cd ~/ResearchPaper/training_digitalocean
# And using correct data-dir
python do_train.py --dataset hcp --categorical-dim 8 --trial 1 --data-dir ../data
```
