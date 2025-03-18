import torch
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_DIR = Path(__file__).resolve().parent / ".."