export default [
  { id: 0, 
    name: "1.1_create_data",
    code: [
      { id: 1, 
        input: 
`import torch
torch.__version__`,
        output: 
`'2.4.0+cu121'`,
      }, 
      { id: 2, 
        input: 
`# check if GPU is available
torch.cuda.is_available()`,
        output: 
``,
      },
      { id: 3, 
        input: 
`# Set Device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
device`,
        output: 
`cpu`,
      },
      { id: 4, 
        input: 
`# Count number of devices
torch.cuda.device_count()`,
        output: 
`0`,
      },
      { id: 5,
        input: 
`from sklearn.datasets import make_circles

# 1)Create DATA (circles distribution)
X, y = make_circles(n_samples=1_000,
                    factor=0.7, # Scale factor (0 ~ 1)
                    noise=0.05, # Standard deviation of Gaussian noise
                    random_state=42) # Reproducibility`,
        output: 
``,
      },
      { id: 7,
        input: 
`print(f"First 5 samples of X: \n {X[:5]}")
print(f"First 5 samples of y: \n {y[:5]}")`,
        output: 
`First 5 samples of X: 
 [[ 0.64566871  0.22060161]
 [-0.63536371  0.15242792]
 [-0.73064725  0.20942567]
 [-0.38655766  0.58174748]
 [ 0.44560223 -0.89493556]]
First 5 samples of y: 
 [1 1 1 1 0]`,
      },
      { id: 8,
        input: 
`# Make DataFrame of circle data
import pandas as pd
circles = pd.DataFrame({"X1": X[:, 0],
                        "X2": X[:, 1],
                        "label": y})
circles.head(5)`,
        output: 
`         X1	   X2	label
0  0.645669  0.220602	1
1 -0.635364  0.152428	1
2 -0.730647  0.209426	1
3 -0.386558  0.581747	1
4 0.445602  -0.894936	0`,
      },
      { id: 9,
        input: 
`# Check the different labels
circles.label.value_counts()`,
        output: 
`label	count
1	500
0	500
dtype: int64`,
      },
      { id: 10,
        input: 
`# View the first example of features and labels
X_first = X[0]
y_first = y[0]

print(f"The first sample of X: {X_first}")
print(f"The first sample of y: {y_first}")`,
        output: 
`The first sample of X: [0.64566871 0.22060161]
The first sample of y: 1`,
      },
      { id: 11,
        input: 
`type(X), X.dtype`,
        output: 
`(numpy.ndarray, dtype('float64'))`,
      },
      { id: 12,
        input: 
`X[:5], y[:5]`,
        output: 
`(array([[ 0.64566871,  0.22060161],
        [-0.63536371,  0.15242792],
        [-0.73064725,  0.20942567],
        [-0.38655766,  0.58174748],
        [ 0.44560223, -0.89493556]]),
 array([1, 1, 1, 1, 0]))`,
      },
      { id: 13,
        input: 
`# 1)Turn data into tensors
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

X[:5], y[:5]`,
        output: 
`(tensor([[ 0.6457,  0.2206],
         [-0.6354,  0.1524],
         [-0.7306,  0.2094],
         [-0.3866,  0.5817],
         [ 0.4456, -0.8949]]),
 tensor([1., 1., 1., 1., 0.]))`,
      },
      { id: 14,
        input: 
`type(X), type(y), X.dtype, y.dtype`,
        output: 
`(torch.Tensor, torch.Tensor, torch.float32, torch.float32)`,
      },
    ]
  }, 
  { id: 1, 
    name: "1.2_split_data",
    code: [
      { id: 15,
        input: 
`# 2)Split DATA into Training SET and Test SET
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2, # 20% test, 80% train
                                                    random_state= 42) # make the random split reproducible`,
        output: 
``,
      },
      { id: 16,
        input: 
`len(X_train), len(X_test), len(y_train), len(y_test)`,
        output: 
`(800, 200, 800, 200)`,
      },
    ]
  },
  { id: 2,
    name: "1.3_visualize_data",
    code: [
      { id: 6,
        input: 
`# Visualize, visualize, visualize
import matplotlib.pyplot as plt
plt.scatter(x=X[:, 0],
            y=X[:, 1],
            c=y,
            cmap=plt.cm.PuOr)`,
        output: 
`<matplotlib.collections.PathCollection at 0x7a3ff5c56170>`,
      },
    ]
  },
  { id: 3, 
    name: "2_build_a_model",
    code: [
      { id: 17,
        input: 
`from torch import nn

# 1)Build a model that inherit from nn.Module
class CircleModelV0(nn.Module):
  def __init__(self):
    super().__init__()
    # 1.1)Create 2 linear layers capable of handling input and output features
    self.layer_1 = nn.Linear(in_features=2, out_features=5) # takes in 2 features(X), produces 5 features(neurons)
    self.layer_2 = nn.Linear(in_features=5, out_features=1) # takes in 5 features(neurons), produces 1 features(y)

  # 1.2)Overwrite the forward()
  def forward(self, X):
    return self.layer_2(self.layer_1(X)) # X -> layer_1 -> layer_2

# 2)Create an instance of the model and send it to target device
model_0 = CircleModelV0().to(device)
model_0`,
        output: 
`CircleModelV0(
  (layer_1): Linear(in_features=2, out_features=5, bias=True)
  (layer_2): Linear(in_features=5, out_features=1, bias=True)
)`,
      },
      { id: 18,
        input: 
`next(model_0.parameters()).device`,
        output: 
`device(type='cpu')`,
      },
      { id: 19,
        input: 
`model_0.state_dict()`,
        output: 
`OrderedDict([('layer_1.weight',
              tensor([[ 0.5406,  0.5869],
                      [-0.1657,  0.6496],
                      [-0.1549,  0.1427],
                      [-0.3443,  0.4153],
                      [ 0.6233, -0.5188]])),
             ('layer_1.bias',
              tensor([0.6146, 0.1323, 0.5224, 0.0958, 0.3410])),
             ('layer_2.weight',
              tensor([[-0.0631,  0.3448,  0.0661, -0.2088,  0.1140]])),
             ('layer_2.bias', tensor([-0.2060]))])`,
      },
    ]
  },
  { id: 4,
    name: "3.1_train_a_model",
    code: [
      { id: 20,
        input: 
`# Pick up a loss function
# loss_fn = nn.BCELoss()
loss_fn = nn.BCEWithLogitsLoss() # BCEWithLogitsLoss: BCELOSS with Sigmoid function

# Pick up optimizer
optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.1)`,
        output: 
``
      },
      { id: 21,
        input: 
`# Define a function to calculate the accuray
def accuracy_fn(y_true, y_pred):
  correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where 2 tensors are equal
  acc = (correct/len(y_pred)) * 100
  return acc`,
        output: 
``
      },
      { id: 22,
        input: 
`torch.manual_seed(69)

# Set the number of epochs
epochs = 100

# Put data to target device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

# Track different values
epoch_count = []
train_loss_values = []
test_loss_values = []

# Build training and evaluation loop
for epoch in range(epochs):
  # Training
  model_0.train()

  # 1. Forward pass (model outputs raw logits)
  y_logits = model_0(X_train).squeeze()
  y_pred = torch.round(torch.sigmoid(y_logits))

  # 2. Calculate loss/accuracy
  # loss = loss_fn(torch.sigmoid(y_logits), # Using nn.BCELoss you need torch.sigmoid()
  #                y_train)
  loss = loss_fn(y_logits, # RAW Logits
                 y_train)
  acc = accuracy_fn(y_true=y_train,
                    y_pred=y_pred)

  # 3. Optimizer zero grad
  optimizer.zero_grad()

  # 4. Loss backwards
  loss.backward()

  # 5. Optimizer step
  optimizer.step()

  ### Testing
  model_0.eval()
  with torch.inference_mode():
    # 1. Forward pass
    test_logits = model_0(X_test).squeeze()
    test_pred = torch.round(torch.sigmoid(test_logits))
    # 2. Calculate loss/accuracy
    test_loss = loss_fn(test_logits,
                        y_test)
    test_acc = accuracy_fn(y_true=y_test,
                           y_pred=test_pred)

  # Print out what's happening every 10 epochs
  if epoch % 10 == 0:
    epoch_count.append(epoch)
    train_loss_values.append(loss)
    test_loss_values.append(test_loss)
    print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss: .5f}, Test acc: {test_acc:.2f}%")`,
        output: 
`Epoch: 0 | Loss: 0.69567, Accuracy: 50.00% | Test loss:  0.69720, Test acc: 50.00%
Epoch: 10 | Loss: 0.69400, Accuracy: 49.88% | Test loss:  0.69615, Test acc: 49.50%
Epoch: 20 | Loss: 0.69339, Accuracy: 43.00% | Test loss:  0.69587, Test acc: 44.50%
Epoch: 30 | Loss: 0.69317, Accuracy: 47.62% | Test loss:  0.69581, Test acc: 45.00%
Epoch: 40 | Loss: 0.69309, Accuracy: 49.12% | Test loss:  0.69579, Test acc: 47.00%
Epoch: 50 | Loss: 0.69305, Accuracy: 49.62% | Test loss:  0.69577, Test acc: 46.50%
Epoch: 60 | Loss: 0.69303, Accuracy: 49.88% | Test loss:  0.69574, Test acc: 46.00%
Epoch: 70 | Loss: 0.69302, Accuracy: 50.62% | Test loss:  0.69570, Test acc: 46.50%
Epoch: 80 | Loss: 0.69301, Accuracy: 50.25% | Test loss:  0.69566, Test acc: 46.50%
Epoch: 90 | Loss: 0.69300, Accuracy: 50.50% | Test loss:  0.69562, Test acc: 46.50%`
      },
    ]
  }, 
  { id: 5, 
    name: "3.2_test_a_model",
    code: [
      { id: 23,
        input: 
`import torch
import matplotlib.pyplot as plt
import numpy as np

# Define a Decison Boundary function to visualize model
def plot_decision_boundary(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor):
    """Plots decision boundaries of model predicting on X in comparison to y.

    Source - https://madewithml.com/courses/foundations/neural-networks/ (with modifications)
    """
    # Put everything to CPU (works better with NumPy + Matplotlib)
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    # Setup prediction boundaries and grid
    x_min, x_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2
    y_min, y_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    # Make features
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # Make predictions for binary classification
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)
        y_pred = torch.round(torch.sigmoid(y_logits))

    # Reshape preds and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.PuOr, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.PuOr)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())`,
        output: 
``
      },
      { id: 24,
        input: 
`# Plot decision boundaries for Training and Test SETs
plt.figure(figsize=(9, 4))
plt.subplot(1, 2, 1) # subplot(nrows, ncols, index)
plt.title("TRAINING")
plot_decision_boundary(model_0, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("TEST")
plot_decision_boundary(model_0, X_test, y_test)`,
        output: 
``
      },
    ]
  },
  { id: 6, 
    name: "3.3_check_loss_curves",
    code: [
      { id: 25,
        input: 
`import numpy as np
train_loss_values = np.array(torch.tensor(train_loss_values).numpy())
test_loss_values = np.array(torch.tensor(test_loss_values).numpy())
train_loss_values, test_loss_values`,
        output: 
`(array([0.6956717 , 0.69399756, 0.6933943 , 0.6931729 , 0.6930874 ,
        0.6930506 , 0.6930317 , 0.69302   , 0.6930114 , 0.6930044 ],
       dtype=float32),
 array([0.69720024, 0.6961511 , 0.6958704 , 0.6958076 , 0.69579   ,
        0.6957708 , 0.69574153, 0.69570416, 0.6956622 , 0.69561833],
       dtype=float32))`
      },
      { id: 26,
        input: 
`# Plot the loss curves
plt.plot(epoch_count, train_loss_values, label="Train LOSS")
plt.plot(epoch_count, test_loss_values, label="Test LOSS")
plt.title("Training and Testing LOSS Curves")
plt.ylabel("LOSS")
plt.xlabel("Epochs")
plt.legend()`,
        output: 
`<matplotlib.legend.Legend at 0x7fc12a79ca90>`
      },
    ]
  },
  { id: 7, 
    name: "4.1.1_build_a_better_model",
    code: [
      { id: 27,
        input: 
`class CircleModelV1(nn.Module):
  def __init__(self):
    super().__init__()
    self.layer_1 = nn.Linear(in_features=2, out_features=10)
    self.layer_2 = nn.Linear(in_features=10, out_features=1)
    self.relu = nn.ReLU()

  def forward(self, x):
    return self.layer_2(self.relu(self.layer_1(x)))
    # this way of writing operations leverages speedups where possible behind the scene

model_1 = CircleModelV1().to(device)
model_1`,
        output: 
`CircleModelV1(
  (layer_1): Linear(in_features=2, out_features=10, bias=True)
  (layer_2): Linear(in_features=10, out_features=1, bias=True)
  (relu): ReLU()
)`,
      },
    ]
  },
  { id: 8, 
    name: "4.1.2_train_test_model",
    code: [
      { id: 28,
        input: 
`# Setup loss and optimizer
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model_1.parameters(), lr=0.1)`,
        output: 
``,
      },
      { id: 29,
        input: 
`# Fit the model
torch.manual_seed(69)
epochs = 100

# Put all data on target device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

# Track different values
epoch_count = []
train_loss_values = []
test_loss_values = []

for epoch in range(epochs):
  # 1. Forward pass
  y_logits = model_1(X_train).squeeze()
  y_pred = torch.round(torch.sigmoid(y_logits))

  # 2. Calculate loss and accuracy
  loss = loss_fn(y_logits, y_train)
  acc = accuracy_fn(y_true=y_train,
                    y_pred=y_pred)

  # 3. Optimizer zero grad
  optimizer.zero_grad()

  # 4. Loss backward
  loss.backward()

  # 5. Optimizer step
  optimizer.step()

  ### Testing
  model_1.eval()
  with torch.inference_mode():
    # 1. Forward pass
    test_logits = model_1(X_test).squeeze()
    test_pred = torch.round(torch.sigmoid(test_logits))
    # 2. Calculate the loss and accuracy
    test_loss = loss_fn(test_logits, y_test)
    test_acc = accuracy_fn(y_true=y_test,
                           y_pred=test_pred)

  # Print out what's happening
  if epoch % 10 == 0:
    epoch_count.append(epoch)
    train_loss_values.append(loss)
    test_loss_values.append(test_loss)
    print(f"Epoch: {epoch} | Loss: {loss:.5f}, accuracy: {acc:.2f} | Test loss: {test_loss: .5f}, Test Accuracy: {test_acc: .2f}")`,
        output: 
`Epoch: 0 | Loss: 0.69779, accuracy: 53.12 | Test loss:  0.69798, Test Accuracy:  54.00
Epoch: 10 | Loss: 0.69540, accuracy: 52.75 | Test loss:  0.69533, Test Accuracy:  56.50
Epoch: 20 | Loss: 0.69362, accuracy: 51.38 | Test loss:  0.69334, Test Accuracy:  55.00
Epoch: 30 | Loss: 0.69214, accuracy: 52.25 | Test loss:  0.69170, Test Accuracy:  55.00
Epoch: 40 | Loss: 0.69083, accuracy: 57.12 | Test loss:  0.69026, Test Accuracy:  54.50
Epoch: 50 | Loss: 0.68962, accuracy: 59.38 | Test loss:  0.68894, Test Accuracy:  55.50
Epoch: 60 | Loss: 0.68845, accuracy: 59.75 | Test loss:  0.68769, Test Accuracy:  56.50
Epoch: 70 | Loss: 0.68731, accuracy: 59.50 | Test loss:  0.68649, Test Accuracy:  57.50
Epoch: 80 | Loss: 0.68618, accuracy: 58.38 | Test loss:  0.68532, Test Accuracy:  58.50
Epoch: 90 | Loss: 0.68508, accuracy: 58.63 | Test loss:  0.68417, Test Accuracy:  57.50`,
      },
      { id: 30,
        input: 
`# Plot decision boundaries for training and test sets
plt.figure(figsize=(9, 4))
plt.subplot(1, 2, 1) # subplot(nrows, ncols, index)
plt.title("TRAINING")
plot_decision_boundary(model_1, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("TEST")
plot_decision_boundary(model_1, X_test, y_test)`,
        output: 
``,
      },
    ]
  },
  { id: 9, 
    name: "4.1.3_plot_loss_curves",
    code: [
      { id: 31,
        input: 
`import numpy as np
train_loss_values = np.array(torch.tensor(train_loss_values).numpy())
test_loss_values = np.array(torch.tensor(test_loss_values).numpy())

# Plot the loss curves
plt.plot(epoch_count, train_loss_values, label="Train LOSS")
plt.plot(epoch_count, test_loss_values, label="Test LOSS")
plt.title("Training and Testing LOSS Curves")
plt.ylabel("LOSS")
plt.xlabel("Epochs")
plt.legend()`,
        output: 
`<matplotlib.legend.Legend at 0x7fc130192b90>`,
      },
    ]
  },
  { id: 10, 
    name: "4.2.1_build_a_new_model",
    code: [
      { id: 27,
        input: 
`from torch import nn

class CircleModelV2(nn.Module):
  def __init__(self):
    super().__init__()
    self.layer_1 = nn.Linear(in_features=2, out_features=10)
    self.layer_2 = nn.Linear(in_features=10, out_features=1)
    self.relu = nn.ReLU()

  def forward(self, x):
    return self.layer_2(self.relu(self.layer_1(x)))

model_2 = CircleModelV2().to(device)
model_2`,
        output: 
`CircleModelV2(
  (layer_1): Linear(in_features=2, out_features=10, bias=True)
  (layer_2): Linear(in_features=10, out_features=1, bias=True)
  (relu): ReLU()
)`,
      },
    ]
  },
  { id: 11, 
    name: "4.2.2_train_test_model",
    code: [
      { id: 28,
        input: 
`# Pick a loss function and optimizer
loss_fn = nn.BCEWithLogitsLoss() # Sigmoid built-in

optimizer = torch.optim.SGD(params=model_2.parameters(),
                            lr=0.1)`,
        output: 
``,
      },
      { id: 29,
        input: 
`torch.manual_seed(69)

epochs = 1000

# Put data to the target device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

# Track different values
epoch_count = []
train_loss_values = []
test_loss_values = []

# Build training and evaluation loop
for epoch in range(epochs):
  ### TRAINING
  model_2.train()
  # 1)Forward pass (model outputs raw logits)
  y_logits = model_2(X_train).squeeze()
  y_pred = torch.round(torch.sigmoid(y_logits))

  # 2)Calculate LOSS and Accuracy
  loss = loss_fn(y_logits, y_train)
  acc = accuracy_fn(y_true=y_train, y_pred=y_pred)
  optimizer.zero_grad() # 3)Optimizer ZERO Gradient
  loss.backward() # 4)Loss backwards (backPropagation)
  optimizer.step() # 5)Optimizer step

  ### TESTING
  model_2.eval()
  with torch.inference_mode():
    # 1)Forward pass
    test_logits = model_2(X_test).squeeze()
    test_pred = torch.round(torch.sigmoid(test_logits))
    # 2)Calculate loss/accuracy
    test_loss = loss_fn(test_logits, y_test)
    test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)

  # Print out what's happening every 10 epochs
  if epoch % 100 == 0:
    epoch_count.append(epoch)
    train_loss_values.append(loss)
    test_loss_values.append(test_loss)
    print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss: .5f}, Test acc: {test_acc:.2f}%")`,
        output: 
`Epoch: 0 | Loss: 0.69779, Accuracy: 57.50% | Test loss:  0.69798, Test acc: 54.00%
Epoch: 100 | Loss: 0.68400, Accuracy: 57.50% | Test loss:  0.68305, Test acc: 58.00%
Epoch: 200 | Loss: 0.67339, Accuracy: 57.50% | Test loss:  0.67221, Test acc: 60.50%
Epoch: 300 | Loss: 0.66225, Accuracy: 57.50% | Test loss:  0.66060, Test acc: 62.00%
Epoch: 400 | Loss: 0.65013, Accuracy: 57.50% | Test loss:  0.64791, Test acc: 66.00%
Epoch: 500 | Loss: 0.63634, Accuracy: 57.50% | Test loss:  0.63354, Test acc: 70.50%
Epoch: 600 | Loss: 0.61991, Accuracy: 57.50% | Test loss:  0.61700, Test acc: 74.50%
Epoch: 700 | Loss: 0.59961, Accuracy: 57.50% | Test loss:  0.59727, Test acc: 80.00%
Epoch: 800 | Loss: 0.57379, Accuracy: 57.50% | Test loss:  0.57341, Test acc: 85.00%
Epoch: 900 | Loss: 0.54110, Accuracy: 57.50% | Test loss:  0.54444, Test acc: 89.00%`,
      },
      { id: 30,
        input: 
`# Plot decision boundaries for training and test sets
plt.figure(figsize=(9, 4))
plt.subplot(1, 2, 1) # subplot(nrows, ncols, index)
plt.title("TRAINING")
plot_decision_boundary(model_2, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("TEST")
plot_decision_boundary(model_2, X_test, y_test)`,
        output: 
``,
      },
    ]
  },
  { id: 12, 
    name: "4.2.3_plot_loss_curves",
    code: [
      { id: 31,
        input: 
`import numpy as np
train_loss_values = np.array(torch.tensor(train_loss_values).numpy())
test_loss_values = np.array(torch.tensor(test_loss_values).numpy())

# Plot the loss curves
plt.plot(epoch_count, train_loss_values, label="Train LOSS")
plt.plot(epoch_count, test_loss_values, label="Test LOSS")
plt.title("Training and Testing LOSS Curves")
plt.ylabel("LOSS")
plt.xlabel("Epochs")
plt.legend()`,
        output: 
`<matplotlib.legend.Legend at 0x7fc1504d6d10>`,
      },
    ]
  },
  { id: 13, 
    name: "4.3.1_build_a_new_model",
    code: [
      { id: 27,
        input: 
`from torch import nn

class CircleModelV3(nn.Module):
  def __init__(self):
    super().__init__()
    self.layer_1 = nn.Linear(in_features=2, out_features=10)
    self.layer_2 = nn.Linear(in_features=10, out_features=1)
    self.relu = nn.ReLU()

  def forward(self, x):
    return self.layer_2(self.relu(self.layer_1(x)))

model_3 = CircleModelV3().to(device)
model_3`,
        output: 
`CircleModelV3(
  (layer_1): Linear(in_features=2, out_features=10, bias=True)
  (layer_2): Linear(in_features=10, out_features=1, bias=True)
  (relu): ReLU()
)`,
      },
    ]
  },
  { id: 14, 
    name: "4.3.2_train_test_model",
    code: [
      { id: 28,
        input: 
`# Pick a loss function and optimizer
loss_fn = nn.BCEWithLogitsLoss() # Sigmoid built-in
optimizer = torch.optim.SGD(params=model_3.parameters(),
                            lr=0.3)`,
        output: 
``,
      },
      { id: 29,
        input: 
`torch.manual_seed(69)

epochs = 1000

# Put data to the target device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

# Track different values
epoch_count = []
train_loss_values = []
test_loss_values = []

# Build training and evaluation loop
for epoch in range(epochs):
  ### TRAINING
  model_3.train()
  # 1)Forward pass (model outputs raw logits)
  y_logits = model_3(X_train).squeeze()
  # 2)Calculate LOSS and Accuracy
  loss = loss_fn(y_logits, y_train)
  acc = accuracy_fn(y_true=y_train, y_pred=y_pred)
  optimizer.zero_grad() # 3)Optimizer ZERO Gradient
  loss.backward() # 4)Loss backwards (backPropagation)
  optimizer.step() # 5)Optimizer step

  ### TESTING
  model_3.eval()
  with torch.inference_mode():
    # 1)Forward pass
    test_logits = model_3(X_test).squeeze()
    test_pred = torch.round(torch.sigmoid(test_logits))
    # 2)Calculate loss/accuracy
    test_loss = loss_fn(test_logits, y_test)
    test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)

  # Print out what's happening every 10 epochs
  if epoch % 100 == 0:
    epoch_count.append(epoch)
    train_loss_values.append(loss)
    test_loss_values.append(test_loss)
    print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss: .5f}, Test acc: {test_acc:.2f}%")`,
        output: 
`Epoch: 0 | Loss: 0.69779, Accuracy: 57.50% | Test loss:  0.69736, Test acc: 56.00%
Epoch: 100 | Loss: 0.66231, Accuracy: 57.50% | Test loss:  0.66041, Test acc: 62.00%
Epoch: 200 | Loss: 0.62008, Accuracy: 57.50% | Test loss:  0.61680, Test acc: 74.50%
Epoch: 300 | Loss: 0.54165, Accuracy: 57.50% | Test loss:  0.54424, Test acc: 89.00%
Epoch: 400 | Loss: 0.41856, Accuracy: 57.50% | Test loss:  0.44522, Test acc: 93.00%
Epoch: 500 | Loss: 0.29038, Accuracy: 57.50% | Test loss:  0.32788, Test acc: 96.50%
Epoch: 600 | Loss: 0.19058, Accuracy: 57.50% | Test loss:  0.22930, Test acc: 99.00%
Epoch: 700 | Loss: 0.13140, Accuracy: 57.50% | Test loss:  0.16752, Test acc: 100.00%
Epoch: 800 | Loss: 0.09762, Accuracy: 57.50% | Test loss:  0.13082, Test acc: 100.00%
Epoch: 900 | Loss: 0.07669, Accuracy: 57.50% | Test loss:  0.10792, Test acc: 100.00%`,
      },
      { id: 30,
        input: 
`# Plot decision boundaries for training and test sets
plt.figure(figsize=(9, 4))
plt.subplot(1, 2, 1) # subplot(nrows, ncols, index)
plt.title("TRAINING")
plot_decision_boundary(model_3, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("TEST")
plot_decision_boundary(model_3, X_test, y_test)`,
        output: 
``,
      },
    ]
  },
  { id: 15, 
    name: "4.3.3_plot_loss_curves",
    code: [
      { id: 31,
        input: 
`import numpy as np
train_loss_values = np.array(torch.tensor(train_loss_values).numpy())
test_loss_values = np.array(torch.tensor(test_loss_values).numpy())

# Plot the loss curves
plt.plot(epoch_count, train_loss_values, label="Train LOSS")
plt.plot(epoch_count, test_loss_values, label="Test LOSS")
plt.title("Training and Testing LOSS Curves")
plt.ylabel("LOSS")
plt.xlabel("Epochs")
plt.legend()`,
        output: 
`<matplotlib.legend.Legend at 0x7fc130584fd0>`,
      },
    ]
  },
  { id: 16, 
    name: "5.0_save_model_choose",
    code: [
      { id: 32,
        input: 
`# Plot decision boundaries for training and test sets
plt.figure(figsize=(14, 3))
plt.subplot(1, 4, 1) # subplot(nrows, ncols, index)
plt.title("model_0")
plot_decision_boundary(model_0, X_test, y_test)
plt.subplot(1, 4, 2)
plt.title("model_1")
plot_decision_boundary(model_1, X_test, y_test)
plt.subplot(1, 4, 3)
plt.title("model_2")
plot_decision_boundary(model_2, X_test, y_test)
plt.subplot(1, 4, 4)
plt.title("model_3")
plot_decision_boundary(model_3, X_test, y_test)`,
        output: 
``,
      }
    ]
  },
  { id: 17, 
    name: "5.1_save_model_model_3",
    code: [
      { id: 33,
        input: 
`from pathlib import Path

# 1)Create models directory
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2)Create model save path
MODEL_NAME = "01_aBinaryClassification_model_3.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3)Save the model state dict
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_3.state_dict(), f=MODEL_SAVE_PATH)`,
        output: 
``,
      },
      { id: 34,
        input: 
`model_3.state_dict()`,
        output:
`OrderedDict([('layer_1.weight',
              tensor([[ 1.4683,  3.8127],
                      [-0.4738, -0.0365],
                      [ 1.1301, -1.0694],
                      [-2.4541, -0.5457],
                      [ 1.9168,  0.9381],
                      [ 1.9147, -2.4138],
                      [ 1.0443, -2.0134],
                      [ 3.2434, -1.4374],
                      [-3.0379,  0.0088],
                      [-0.9749, -3.3199]])),
             ('layer_1.bias',
              tensor([-0.6864,  0.9496,  3.3581, -0.1093, -0.2119, -0.2084, -0.2104, -0.5631,
                      -0.1237, -1.1244])),
             ('layer_2.weight',
              tensor([[-4.0986,  0.6790,  3.6823, -2.3544, -2.1186, -2.9299, -2.1206, -3.4726,
                       -3.0189, -3.5742]])),
             ('layer_2.bias', tensor([2.1943]))])`,
      },
    ]
  },
  { id: 18, 
    name: "5.0_save_model_load",
    code: [
      { id: 35,
        input: 
`# 1)Instantiate a new model
loaded_model_0 = CircleModelV3()

# 2)Load the state_dict of model (update the training weights)
loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))`,
        output: 
`<ipython-input-104-9d5f24ceb354>:5: FutureWarning: You are using \`torch.load\` with \`weights_only=False\` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for \`weights_only\` will be flipped to \`True\`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via \`torch.serialization.add_safe_globals\`. We recommend you start setting \`weights_only=True\` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
<All keys matched successfully>`,
      },
      { id: 36,
        input: 
`loaded_model_0.state_dict()`,
        output: 
`OrderedDict([('layer_1.weight',
              tensor([[ 1.4683,  3.8127],
                      [-0.4738, -0.0365],
                      [ 1.1301, -1.0694],
                      [-2.4541, -0.5457],
                      [ 1.9168,  0.9381],
                      [ 1.9147, -2.4138],
                      [ 1.0443, -2.0134],
                      [ 3.2434, -1.4374],
                      [-3.0379,  0.0088],
                      [-0.9749, -3.3199]])),
             ('layer_1.bias',
              tensor([-0.6864,  0.9496,  3.3581, -0.1093, -0.2119, -0.2084, -0.2104, -0.5631,
                      -0.1237, -1.1244])),
             ('layer_2.weight',
              tensor([[-4.0986,  0.6790,  3.6823, -2.3544, -2.1186, -2.9299, -2.1206, -3.4726,
                       -3.0189, -3.5742]])),
             ('layer_2.bias', tensor([2.1943]))])`,
      },
      { id: 37,
        input: 
`# Make predictions with loaded model
loaded_model_0.eval()
with torch.inference_mode():
  test_logits = loaded_model_0(X_test).squeeze()
  loaded_model_preds = torch.round(torch.sigmoid(test_logits))`,
        output: 
``,
      },
      { id: 38,
        input: 
`# Make predictions with model_3
model_3.eval()
with torch.inference_mode():
  test_logits = model_3(X_test).squeeze()
  y_preds = torch.round(torch.sigmoid(test_logits))`,
        output: 
``,
      },
      { id: 39,
        input: 
`# 3)Compare the predictions of loaded model and original model
y_preds[:69] == loaded_model_preds[:69]`,
        output: 
`tensor([True, True, True, True, True, True, True, True, True, True, True, True,
        True, True, True, True, True, True, True, True, True, True, True, True,
        True, True, True, True, True, True, True, True, True, True, True, True,
        True, True, True, True, True, True, True, True, True, True, True, True,
        True, True, True, True, True, True, True, True, True, True, True, True,
        True, True, True, True, True, True, True, True, True])`,
      },
    ]
  },
]