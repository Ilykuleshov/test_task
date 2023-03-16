
import torch
import glob
from pathlib import Path
from argparse import ArgumentParser
import zipfile
import tempfile

from cifar_simclr_resnet.const import REPO_ROOT

def extract_weights():
    file = next((REPO_ROOT / 'weights/raw').iterdir())
    with tempfile.TemporaryDirectory() as tempdir, zipfile.ZipFile(file, 'r') as zfile:
        zfile.extractall(tempdir)
        weights_file = glob.glob('*.pth*', root_dir=tempdir)[0]
        weights = torch.load(Path(tempdir) / weights_file)
    
    return {k[len('backbone.'):]: v for k, v in weights['state_dict'].items()}


if __name__ == '__main__':
    torch.save(extract_weights(), REPO_ROOT / 'weights/weights.pt')
