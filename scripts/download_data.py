from torchvision.datasets.cifar import CIFAR10
from cifar_simclr_resnet.const import REPO_ROOT

if __name__ == '__main__':
    CIFAR10(REPO_ROOT / 'data', train=False, download=True)
    CIFAR10(REPO_ROOT / 'data', train=True, download=True)
