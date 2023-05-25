from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

train = datasets.MNIST(root='./data', train=True, download=True, transform=ToTensor())
dataset = DataLoader(train, batch_size=32)

class imageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*(28-6)*(28-6), 10)
        )

    def forward(self, x):
        return self.model(x)
    
clf = imageClassifier().to('cuda')
opt = Adam(clf.parameters(), lr=1e-3)
lossFn = nn.CrossEntropyLoss()

if __name__ == '__main__':
    for epoch in range(10):
        for batch in dataset:
            X, y = batch
            X, y = X.to('cuda'), y.to('cuda')
            yhat = clf(X)
            loss = lossFn(yhat, y)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
        print(f'Epoch {epoch+1} loss: {loss.item()}')
    
    with open('model.pt', 'wb') as f:
        save(clf.state_dict(), f)
