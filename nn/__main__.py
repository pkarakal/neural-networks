import sys
import numpy as np
import torch
from torchvision import transforms
from nn.utils import download_mnist_dataset, visualize_batch, visualize_image, train, evaluate, stopwatch
from torch import nn
from nn.MNIST import MNIST
from nn.cli_parser import Parser

if __name__ == "__main__":
    parser = Parser()
    cli = parser.parseCLI(sys.argv[1:])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 0
    pin_memory = False

    if device == "cuda":
        num_workers = 1
        pin_memory = True
    print(f"Will be using {device} for training and testing")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    batch_size = cli.get('batch_size')
    n_epochs = cli.get('epochs')

    train_set = download_mnist_dataset('mnist', True, transform=transform)
    test_set = download_mnist_dataset('mnist', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, num_workers=num_workers,
                                               pin_memory=pin_memory)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True, num_workers=num_workers,
                                              pin_memory=pin_memory)
    # visualizes a part of the batch
    visualize_batch(train_loader)
    # visualizes an image in more detail
    visualize_image(np.squeeze((next(iter(train_loader))[0])[0].numpy()))
    layers = cli.get("layers")
    if layers > 4:
        print(f"{layers} is greater than 4. Going to use 4 hidden layers")
    mnist_model = MNIST(28 * 28, hidden_size1=512, hidden_size2=512, hidden_size3=128 if layers > 2 else None,
                        hidden_size4=64 if layers > 3 else None, dropout_rate=0.2, output=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mnist_model.parameters(), lr=0.002)
    images, labels = next(iter(train_loader))
    images = images.view(images.shape[0], -1)
    mnist_model.to(device)
    print(f"Training took {stopwatch(lambda: train(model=mnist_model, device=device, train_loader=train_loader, criterion=criterion, optimizer=optimizer, epochs=n_epochs))}s")
    print(f"Evaluation took {stopwatch(lambda: evaluate(model=mnist_model, device=device, test_loader=test_loader, criterion=criterion, labels=labels))}s")
    if cli.get('save_model'):
        torch.save(mnist_model.state_dict(), "mnist.pt")
