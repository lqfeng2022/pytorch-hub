export default [
  { id: 0, 
    name: "1.1_create_data",
    code: [
      // Regular setting
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
      // 1.1 Create Data
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
      { id: 6,
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
      { id: 7,
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
      { id: 8,
        input: 
`# Check the different labels
circles.label.value_counts()`,
        output: 
`label	count
1	500
0	500
dtype: int64`,
      },
      { id: 9,
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
      { id: 10,
        input: 
`type(X), X.dtype`,
        output: 
`(numpy.ndarray, dtype('float64'))`,
      },
      { id: 11,
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
      { id: 12,
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
      { id: 13,
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
      { id: 14,
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
      { id: 15,
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
      { id: 16,
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
              tensor([[ 0.2291, -0.6847],
                      [ 0.0437,  0.5737],
                      [ 0.1430,  0.1516],
                      [ 0.0209,  0.3754],
                      [ 0.0614, -0.1734]], device='cuda:0')),
             ('layer_1.bias',
              tensor([-0.2749,  0.2505, -0.1695, -0.3641,  0.4621], device='cuda:0')),
             ('layer_2.weight',
              tensor([[ 0.3347,  0.1223, -0.1030, -0.4103,  0.3121]], device='cuda:0')),
             ('layer_2.bias', tensor([0.2674], device='cuda:0'))])`,
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

  # 1)Forward pass (model outputs raw logits)
  y_logits = model_0(X_train).squeeze()
  y_pred = torch.round(torch.sigmoid(y_logits))
  # 2)Calculate loss/accuracy
  loss = loss_fn(y_logits, y_train) # RAW Logits
  acc = accuracy_fn(y_true=y_train, y_pred=y_pred)
  # 3)Optimizer zero grad
  optimizer.zero_grad()
  # 4)Loss backwards
  loss.backward()
  # 5)Optimizer step
  optimizer.step()

  ### Testing
  model_0.eval()
  with torch.inference_mode():
    # 1)Forward pass
    test_logits = model_0(X_test).squeeze()
    test_pred = torch.round(torch.sigmoid(test_logits))
    # 2)Calculate loss/accuracy
    test_loss = loss_fn(test_logits, y_test)
    test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)

  # Print out loss and accuracy every 10 epochs
  if epoch % 10 == 0:
    epoch_count.append(epoch)
    train_loss_values.append(loss)
    test_loss_values.append(test_loss)
    print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss: .5f}, Test acc: {test_acc:.2f}%")`,
        output: 
`Epoch: 0 | Loss: 0.73573, Accuracy: 50.00% | Test loss:  0.71928, Test acc: 50.00%
Epoch: 10 | Loss: 0.71432, Accuracy: 55.38% | Test loss:  0.70242, Test acc: 57.00%
Epoch: 20 | Loss: 0.70521, Accuracy: 58.63% | Test loss:  0.69586, Test acc: 62.50%
Epoch: 30 | Loss: 0.70089, Accuracy: 54.25% | Test loss:  0.69321, Test acc: 55.00%
Epoch: 40 | Loss: 0.69859, Accuracy: 52.62% | Test loss:  0.69213, Test acc: 52.00%
Epoch: 50 | Loss: 0.69721, Accuracy: 52.38% | Test loss:  0.69171, Test acc: 52.50%
Epoch: 60 | Loss: 0.69629, Accuracy: 51.38% | Test loss:  0.69159, Test acc: 52.00%
Epoch: 70 | Loss: 0.69563, Accuracy: 51.00% | Test loss:  0.69162, Test acc: 50.50%
Epoch: 80 | Loss: 0.69513, Accuracy: 50.88% | Test loss:  0.69171, Test acc: 49.50%
Epoch: 90 | Loss: 0.69474, Accuracy: 50.75% | Test loss:  0.69185, Test acc: 50.00%`
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

# Define a plot function to visualize model
def plot_decision_boundary(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor):
    """Plots decision boundaries of model predicting on X in comparison to y.

    Source - https://madewithml.com/courses/foundations/neural-networks/ (with modifications)
    """
    # Put everything to CPU (works for NumPy + Matplotlib)
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
`(array([0.7357303 , 0.71431684, 0.7052077 , 0.70088726, 0.6985858 ,
        0.69720733, 0.6962918 , 0.6956338 , 0.69513494, 0.69474363],
       dtype=float32),
 array([0.7192759 , 0.7024158 , 0.6958596 , 0.69320744, 0.6921252 ,
        0.6917087 , 0.6915907 , 0.6916151 , 0.69171184, 0.6918471 ],
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
    # This nested way of writing operations takes advantage of behind-the-scenes speedups whenever possible

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
`# Pick up loss function and optimizer
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

# Training loop
for epoch in range(epochs):
  # 1)Forward pass
  y_logits = model_1(X_train).squeeze()
  y_pred = torch.round(torch.sigmoid(y_logits))

  # 2)Calculate loss and accuracy
  loss = loss_fn(y_logits, y_train)
  acc = accuracy_fn(y_true=y_train, y_pred=y_pred)
  optimizer.zero_grad() # 3)Optimizer ZERO Gradient
  loss.backward() # 4)Loss backwards (backPropagation)
  optimizer.step() # 5)Optimizer step

  ### Testing
  model_1.eval()
  with torch.inference_mode():
    # 1)Forward pass
    test_logits = model_1(X_test).squeeze()
    test_pred = torch.round(torch.sigmoid(test_logits))
    # 2)Calculate the loss and accuracy
    test_loss = loss_fn(test_logits, y_test)
    test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)

  # Print out the loss and accuracy every 10 epochs
  if epoch % 10 == 0:
    epoch_count.append(epoch)
    train_loss_values.append(loss)
    test_loss_values.append(test_loss)
    print(f"Epoch: {epoch} | Loss: {loss:.5f}, accuracy: {acc:.2f} | Test loss: {test_loss: .5f}, Test Accuracy: {test_acc: .2f}")`,
        output: 
`Epoch: 0 | Loss: 0.69083, accuracy: 53.62 | Test loss:  0.68830, Test Accuracy:  55.50
Epoch: 10 | Loss: 0.68809, accuracy: 53.12 | Test loss:  0.68669, Test Accuracy:  52.00
Epoch: 20 | Loss: 0.68634, accuracy: 53.50 | Test loss:  0.68572, Test Accuracy:  53.00
Epoch: 30 | Loss: 0.68494, accuracy: 53.37 | Test loss:  0.68493, Test Accuracy:  53.00
Epoch: 40 | Loss: 0.68370, accuracy: 53.37 | Test loss:  0.68421, Test Accuracy:  53.00
Epoch: 50 | Loss: 0.68252, accuracy: 53.50 | Test loss:  0.68351, Test Accuracy:  53.00
Epoch: 60 | Loss: 0.68138, accuracy: 54.00 | Test loss:  0.68281, Test Accuracy:  53.00
Epoch: 70 | Loss: 0.68027, accuracy: 54.25 | Test loss:  0.68211, Test Accuracy:  53.50
Epoch: 80 | Loss: 0.67918, accuracy: 54.50 | Test loss:  0.68139, Test Accuracy:  53.50
Epoch: 90 | Loss: 0.67811, accuracy: 54.62 | Test loss:  0.68066, Test Accuracy:  54.00`,
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
      { id: 32,
        input: 
`from torch import nn

class CircleModelV2(nn.Module):
  def __init__(self):
    super().__init__()
    self.layer_1 = nn.Linear(in_features=2, out_features=10)
    self.layer_2 = nn.Linear(in_features=10, out_features=1)
    self.relu = nn.ReLU() # add ReLU function here

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
      { id: 33,
        input: 
`# Pick up loss function and optimizer
loss_fn = nn.BCEWithLogitsLoss() # Sigmoid built-in
optimizer = torch.optim.SGD(params=model_2.parameters(),
                            lr=0.1)`,
        output: 
``,
      },
      { id: 34,
        input: 
`torch.manual_seed(69)

epochs = 1000 # 100 -> 1,000

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

  # Print out loss and accurary every 10 epochs
  if epoch % 100 == 0:
    epoch_count.append(epoch)
    train_loss_values.append(loss)
    test_loss_values.append(test_loss)
    print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss: .5f}, Test acc: {test_acc:.2f}%")`,
        output: 
`Epoch: 0 | Loss: 0.69083, Accuracy: 53.62% | Test loss:  0.68830, Test acc: 55.50%
Epoch: 100 | Loss: 0.67704, Accuracy: 55.12% | Test loss:  0.67993, Test acc: 54.50%
Epoch: 200 | Loss: 0.66572, Accuracy: 62.62% | Test loss:  0.67164, Test acc: 62.50%
Epoch: 300 | Loss: 0.65251, Accuracy: 69.38% | Test loss:  0.66123, Test acc: 66.00%
Epoch: 400 | Loss: 0.63537, Accuracy: 73.38% | Test loss:  0.64740, Test acc: 71.50%
Epoch: 500 | Loss: 0.61388, Accuracy: 78.38% | Test loss:  0.62967, Test acc: 74.50%
Epoch: 600 | Loss: 0.58537, Accuracy: 84.50% | Test loss:  0.60625, Test acc: 78.50%
Epoch: 700 | Loss: 0.54921, Accuracy: 90.38% | Test loss:  0.57596, Test acc: 84.00%
Epoch: 800 | Loss: 0.50659, Accuracy: 95.12% | Test loss:  0.53875, Test acc: 90.50%
Epoch: 900 | Loss: 0.45954, Accuracy: 97.88% | Test loss:  0.49689, Test acc: 94.50%`,
      },
      { id: 35,
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
      { id: 36,
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
      { id: 37,
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
      { id: 38,
        input: 
`# Pick up loss function and optimizer
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=model_3.parameters(),
                            lr=0.3)`,
        output: 
``,
      },
      { id: 39,
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

  # Print out loss/accuracy every 10 epochs
  if epoch % 100 == 0:
    epoch_count.append(epoch)
    train_loss_values.append(loss)
    test_loss_values.append(test_loss)
    print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss: .5f}, Test acc: {test_acc:.2f}%")`,
        output: 
`Epoch: 0 | Loss: 0.69083, Accuracy: 98.62% | Test loss:  0.68785, Test acc: 53.00%
Epoch: 100 | Loss: 0.65257, Accuracy: 98.62% | Test loss:  0.66105, Test acc: 66.00%
Epoch: 200 | Loss: 0.58578, Accuracy: 98.62% | Test loss:  0.60608, Test acc: 78.50%
Epoch: 300 | Loss: 0.46037, Accuracy: 98.62% | Test loss:  0.49677, Test acc: 94.50%
Epoch: 400 | Loss: 0.31990, Accuracy: 98.62% | Test loss:  0.36718, Test acc: 97.00%
Epoch: 500 | Loss: 0.21253, Accuracy: 98.62% | Test loss:  0.26066, Test acc: 99.00%
Epoch: 600 | Loss: 0.14532, Accuracy: 98.62% | Test loss:  0.19060, Test acc: 99.50%
Epoch: 700 | Loss: 0.10525, Accuracy: 98.62% | Test loss:  0.14739, Test acc: 100.00%
Epoch: 800 | Loss: 0.08044, Accuracy: 98.62% | Test loss:  0.11945, Test acc: 100.00%
Epoch: 900 | Loss: 0.06447, Accuracy: 98.62% | Test loss:  0.10122, Test acc: 100.00%`,
      },
      { id: 40,
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
      { id: 41,
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
      { id: 42,
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
      { id: 43,
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
`Saving model to: models/01_aBinaryClassification_model_3.pth`,
      },
      { id: 44,
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
      { id: 45,
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
      { id: 46,
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
      { id: 47,
        input: 
`# Make predictions with loaded model
loaded_model_0.to(device) # Remember putting model on GPUs if available

loaded_model_0.eval()
with torch.inference_mode():
  test_logits = loaded_model_0(X_test).squeeze()
  loaded_model_preds = torch.round(torch.sigmoid(test_logits))`,
        output: 
``,
      },
      { id: 48,
        input: 
`# Make predictions with model_3
model_3.to(device) # Put model on the GPUs

model_3.eval()
with torch.inference_mode():
  test_logits = model_3(X_test).squeeze()
  y_preds = torch.round(torch.sigmoid(test_logits))`,
        output: 
``,
      },
      { id: 49,
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