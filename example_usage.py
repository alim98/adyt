import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse

# Import the AdaptiveDynamicTanh module
from dynamic_tanh_adaptive import AdaptiveDynamicTanh, convert_ln_to_adyt, update_adyt_grad_norms

# Simple ConvNet with normalization
class SimpleConvNet(nn.Module):
    def __init__(self, use_adaptive_dyt=False, lambda_factor=0.1, smooth_factor=0.9):
        super(SimpleConvNet, self).__init__()
        self.use_adaptive_dyt = use_adaptive_dyt
        
        # Define the model with LayerNorm
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.ln1 = nn.LayerNorm(128)
        self.fc2 = nn.Linear(128, 10)
        
        # Convert LayerNorm to AdaptiveDynamicTanh if requested
        if use_adaptive_dyt:
            self.ln1 = AdaptiveDynamicTanh(
                128,                     # normalized_shape
                channels_last=True,      # channels last format
                alpha_init_value=0.5,    # initial alpha value
                lambda_factor=lambda_factor,  # lambda factor for adaptation
                smooth_factor=smooth_factor   # smoothing factor for gradient norm
            )
            print(f"Using AdaptiveDynamicTanh with lambda={lambda_factor}, smooth_factor={smooth_factor}")
        else:
            print("Using standard LayerNorm")
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.ln1(x)  # LayerNorm or AdaptiveDynamicTanh
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        
        # Update gradient norms if using AdaptiveDynamicTanh
        if model.use_adaptive_dyt:
            update_adyt_grad_norms(model)
            
        optimizer.step()
        
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    return correct / len(test_loader.dataset)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='AdaptiveDynamicTanh Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--use-adyt', action='store_true', default=False,
                        help='Use AdaptiveDynamicTanh instead of LayerNorm')
    parser.add_argument('--lambda-factor', type=float, default=0.1,
                        help='Lambda factor for AdaptiveDynamicTanh (default: 0.1)')
    parser.add_argument('--smooth-factor', type=float, default=0.9,
                        help='Smooth factor for AdaptiveDynamicTanh (default: 0.9)')
    args = parser.parse_args()
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    try:
        dataset1 = datasets.MNIST('./data', train=True, download=True, transform=transform)
        dataset2 = datasets.MNIST('./data', train=False, transform=transform)
        train_loader = DataLoader(dataset1, **train_kwargs)
        test_loader = DataLoader(dataset2, **test_kwargs)
    except Exception as e:
        print(f"Could not load MNIST dataset: {str(e)}")
        print("Continuing with dummy data...")
        # Create dummy data if MNIST can't be loaded
        class DummyDataset:
            def __init__(self, size):
                self.size = size
                self.data = [(torch.randn(1, 28, 28), torch.randint(0, 10, (1,)).item()) for _ in range(size)]
            
            def __getitem__(self, idx):
                return self.data[idx]
            
            def __len__(self):
                return self.size
        
        train_loader = DataLoader(DummyDataset(10000), **train_kwargs)
        test_loader = DataLoader(DummyDataset(1000), **test_kwargs)

    # Create the model
    model = SimpleConvNet(
        use_adaptive_dyt=args.use_adyt,
        lambda_factor=args.lambda_factor,
        smooth_factor=args.smooth_factor
    ).to(device)
    
    # Alternatively, you can use the conversion function to convert all LayerNorm layers:
    # if args.use_adyt:
    #     model = convert_ln_to_adyt(model, 
    #                               lambda_factor=args.lambda_factor,
    #                               smooth_factor=args.smooth_factor)
    
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.gamma)
    
    # Train and test
    best_acc = 0
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        acc = test(model, device, test_loader)
        if acc > best_acc:
            best_acc = acc
        scheduler.step()

    # Save model if requested
    if args.save_model:
        method = "adyt" if args.use_adyt else "ln"
        torch.save(model.state_dict(), f"mnist_cnn_{method}.pt")
    
    print(f"Best accuracy: {best_acc*100:.2f}%")

if __name__ == '__main__':
    main() 