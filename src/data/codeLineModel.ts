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
`# Create *known* parameters
slope = 0.6
intercept = 0.1

# Create DATA
X = torch.arange(0, 1, 0.02).unsqueeze(dim=1)
y = slope * X + intercept

X[:10], y[:10]`,
        output: 
`(tensor([[0.0000],
         [0.0200],
         [0.0400],
         [0.0600],
         [0.0800],
         [0.1000],
         [0.1200],
         [0.1400],
         [0.1600],
         [0.1800]]),
 tensor([[0.1000],
         [0.1120],
         [0.1240],
         [0.1360],
         [0.1480],
         [0.1600],
         [0.1720],
         [0.1840],
         [0.1960],
         [0.2080]]))`,
      },
      { id: 3,
        input: 
`len(X), len(y)`,
        output:
`(50, 50)`,
      }
    ]
  }, 
  { id: 1, 
    name: "1.2_split_data",
    code: [
      { id: 4, 
        input: 
`# Create a train/test split
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

len(X_train), len(y_train), len(X_test), len(y_test)`,
        output: 
`(40, 40, 10, 10)`,
      }, 
    ]
  }, 
  { id: 2, 
    name: "1.3_visualize_data",
    code: [
      { id: 5,
        input: 
`import matplotlib.pyplot as plt

# Define a plot function to visualize the predictions
def plot_predictions(train_data=X_train,
                     train_labels=y_train,
                     test_data =X_test,
                     test_labels=y_test,
                     currents=None,
                     predictions=None):
  """
  Plots Training DATA and Testing DATA, then compares predictions.
  """
  plt.figure(figsize=(10, 7))

  # Plot Training DATA in blue
  plt.scatter(train_data, train_labels, c="b", s=8, label="Training DATA")

  # Plot Testing DATA in green
  plt.scatter(test_data, test_labels, c="g", s=8, label="Test DATA")

  # To see the Currrent model
  if currents is not None:
    plt.scatter(train_data, currents, c="r", s=8, label="Current DATA")

  # Are there predictions?
  if predictions is not None:
    # Plot the predictions in red (made on the Testing DATA)
    plt.scatter(test_data, predictions, c="r", s=8, label="Predictions")

  # Show the legend
  plt.legend(prop={"size": 15}); # Define the size of scatter labels`,
        output: 
``,
      }, 
      { id: 6,
        input: 
`plot_predictions()`,
        output:
``, // here we need add an image
      }
    ]
  }, 
  { id: 3, 
    name: "2.1_build_a_model",
    code: [
      { id: 7, 
        input: 
`from torch import nn
# 'nn' contains all building blocks for graphs

# 1)Create linear regression model class
class LinearRegressionModel(nn.Module): # <- nn.Module: base class for all nn modules
  # 1.1)Initialize the parameters(weight, bias)
  def __init__(self):
    super().__init__()
    self.weight = nn.Parameter(torch.randn(1, # start with a random weight
                                           requires_grad=True, # can be updated via gradient descent
                                           dtype=torch.float)) # float32/float by default in PyTorch
    self.bias = nn.Parameter(torch.randn(1,
                                         requires_grad=True,
                                         dtype=torch.float))

  # 1.2)Overwrite forward() to define the computation of model
  def forward(self, x: torch.Tensor) -> torch.Tensor: # 'x' is the input data
    return self.weight * x + self.bias # a linear formula (y = w•x + b)`,
        output: 
``,
      }, 
      { id: 8, 
        input: 
`# Create a random seed
torch.manual_seed(42)

# Create an instance of the model
model_0 = LinearRegressionModel()

# Check out the parameters
list(model_0.parameters())`,
        output: 
`[Parameter containing:
 tensor([0.3367], requires_grad=True),
 Parameter containing:
 tensor([0.1288], requires_grad=True)]`,
      }, 
      { id: 9, 
        input: 
`# List named parameters
model_0.state_dict()`,
        output: 
`OrderedDict([('weight', tensor([0.3367])), ('bias', tensor([0.1288]))])`,
      }, 
      { id: 10, 
        input: 
`# check the target formula parameters
slope, intercept`,
        output: 
`(0.6, 0.1)`,
      }, 
      { id: 11, 
        input: 
`# plot our model with Testing DATA
with torch.inference_mode():
  y_preds = model_0(X_train)

print(f"Number of prediction samples: {len(y_preds)}")
print(f"Predicted values: \\n{y_preds[:8]}")`,
        output: 
`Number of prediction samples: 40
Predicted values: 
tensor([[0.1288],
        [0.1355],
        [0.1423],
        [0.1490],
        [0.1557],
        [0.1625],
        [0.1692],
        [0.1759]])`,
      }, 
      { id: 12, 
        input: 
`print(f"Number of testing samples: {len(y_train)}")
print(f"Tested values: \\n{y_train[:8]}")`,
        output: 
`Number of testing samples: 40
Tested values: 
tensor([[0.1000],
        [0.1120],
        [0.1240],
        [0.1360],
        [0.1480],
        [0.1600],
        [0.1720],
        [0.1840]])`,
      }, 
    ]
  }, 
  { id: 4, 
    name: "2.2_visualize_model",
    code: [
      { id: 13,
        input:
    `plot_predictions(currents=y_preds)`,
        output:
    ``, // here is a picture
      },
      { id: 14, 
        input: 
`# Check the difference between testing/prediction values
print(f"y_train - Y_preds: \\n{y_train[:8] - y_preds[:8]}")`,
        output: 
`y_train - Y_preds: 
tensor([[-0.0288],
        [-0.0235],
        [-0.0183],
        [-0.0130],
        [-0.0077],
        [-0.0025],
        [ 0.0028],
        [ 0.0081]])`,
      }, 
    ]
  }, 
  { id: 5,
    name: "3.1_build_a_train_loop",
    code: [
      { id: 19,
        input: 
`# Pick up a loss function
loss_fn = nn.MSELoss()

# Pick up an optimizer
optimizer = torch.optim.SGD(params=model_0.parameters(), # optimize the parameters in target model
                            lr=0.2) # learning rate, most important hyperparameter`,
        output: 
``,
      },
      { id: 20,
        input: 
`# Set up epochs, which is a hyper-parameter, one epoch is one loop through the DATA
epochs = 100

# Track different values
epoch_count = []
train_loss_values = []
test_loss_values = []

### Training
# Loop through the DATA
for epoch in range(epochs):
  # 0)Put model in Training MODE (default state of model)
  model_0.train()

  # 1)Forward Pass (on training DATA using the forward() inside)
  y_pred = model_0(X_train)

  # 2)Calculate the LOSS
  loss = loss_fn(y_pred, y_train) # output = nn.L1Loss(input, target)

  # 3)Optimizer ZERO GRAD
  optimizer.zero_grad() # Here we gotta ZERO the Gradient before BackPropagation

  # 4)Perform BackPropagation (on the LOSS with respect to the Parameters of the model)
  loss.backward()

  # 5)Step the Optimizer (perform gradient descent)
  optimizer.step()
  # by default how the optimizer changes will accumulate through the loop so...
  # we have to zero them above in step 3 for the next iteration of the loop..

  ### Evaluating the model (after every Training LOOP)
  # 0)Put the model in Evaluation MODE
  model_0.eval()
  with torch.inference_mode(): # Turn off Gradient Tracking
    # 1)Do the Forward Pass
    test_pred = model_0(X_test)

    # 2)Calculate LOSS on Testing DATA
    test_loss = loss_fn(test_pred, y_test)

  ### Print out what's happening
  if epoch % 10 == 0:
    epoch_count.append(epoch)
    train_loss_values.append(loss)
    test_loss_values.append(test_loss)
    print(f"Epoch: {epoch} | Train Loss: {loss} | Test loss: {test_loss}")`,
        output: 
`Epoch: 0 | Train Loss: 0.009153849445283413 | Test loss: 0.026034023612737656
Epoch: 10 | Train Loss: 0.0020564072765409946 | Test loss: 0.00869158748537302
Epoch: 20 | Train Loss: 0.001418724306859076 | Test loss: 0.005984040442854166
Epoch: 30 | Train Loss: 0.0009787960443645716 | Test loss: 0.00412844680249691
Epoch: 40 | Train Loss: 0.0006752834306098521 | Test loss: 0.0028482668567448854
Epoch: 50 | Train Loss: 0.00046588649274781346 | Test loss: 0.0019650552421808243
Epoch: 60 | Train Loss: 0.0003214209864381701 | Test loss: 0.0013557180063799024
Epoch: 70 | Train Loss: 0.00022175228514242917 | Test loss: 0.0009353269124403596
Epoch: 80 | Train Loss: 0.00015298996004275978 | Test loss: 0.0006452933885157108
Epoch: 90 | Train Loss: 0.00010554927575867623 | Test loss: 0.0004451951535884291`,
      },
      { id: 21,
        input: 
`train_loss_values, test_loss_values`,
        output: 
`([tensor(0.0092, grad_fn=<MseLossBackward0>),
  tensor(0.0021, grad_fn=<MseLossBackward0>),
  tensor(0.0014, grad_fn=<MseLossBackward0>),
  tensor(0.0010, grad_fn=<MseLossBackward0>),
  tensor(0.0007, grad_fn=<MseLossBackward0>),
  tensor(0.0005, grad_fn=<MseLossBackward0>),
  tensor(0.0003, grad_fn=<MseLossBackward0>),
  tensor(0.0002, grad_fn=<MseLossBackward0>),
  tensor(0.0002, grad_fn=<MseLossBackward0>),
  tensor(0.0001, grad_fn=<MseLossBackward0>)],
 [tensor(0.0260),
  tensor(0.0087),
  tensor(0.0060),
  tensor(0.0041),
  tensor(0.0028),
  tensor(0.0020),
  tensor(0.0014),
  tensor(0.0009),
  tensor(0.0006),
  tensor(0.0004)])`,
      },
    ]
  },
  { id: 6,
    name: "3.2_visualize_trained_model",
    code: [
      { id: 22,
        input: 
`# make predictions with Testing DATA on model_0
with torch.inference_mode():
  y_preds = model_0(X_train)

print(f"Number of prediction samples: {len(y_preds)}")
print(f"Predicted values: \\n{y_preds[:5]}")`,
        output: 
`Number of prediction samples: 40
Predicted values: 
tensor([[0.1151],
        [0.1263],
        [0.1376],
        [0.1488],
        [0.1601]])`
      },
      { id: 23,
        input: 
`plot_predictions(currents=y_preds)`,
        output: 
``
      },
      { id: 24,
        input: 
`# Find our model's learned parameters
print(f"The model learned the following values for weight and bias.")
print(model_0.state_dict())
print("\\nAnd the original values for weight and bias are:")
print(f"weights: {slope}, bias: {intercept}")`,
        output: 
`The model learned the following values for weight and bias.
OrderedDict([('weight', tensor([0.5632])), ('bias', tensor([0.1151]))])

And the original values for weight and bias are:
weights: 0.6, bias: 0.1`
      },
    ]
  },
  { id: 7,
    name: "3.3_test_trained_model",
    code: [
      { id: 25,
        input: 
`plot_predictions(predictions=test_pred)`,
        output: 
``
      }
    ]
  },
  { id: 8,
    name: "3.4_check_the_loss_curves",
    code: [
      { id: 26,
        input: 
`import numpy as np
train_loss_values = np.array(torch.tensor(train_loss_values).numpy())
train_loss_values, test_loss_values`,
        output: 
`(array([0.00915385, 0.00205641, 0.00141872, 0.0009788 , 0.00067528,
        0.00046589, 0.00032142, 0.00022175, 0.00015299, 0.00010555],
       dtype=float32),
 [tensor(0.0260),
  tensor(0.0087),
  tensor(0.0060),
  tensor(0.0041),
  tensor(0.0028),
  tensor(0.0020),
  tensor(0.0014),
  tensor(0.0009),
  tensor(0.0006),
  tensor(0.0004)])`
      },
      { id: 27,
        input: 
`# Plot the loss curves
plt.plot(epoch_count, train_loss_values, label="Train LOSS")
plt.plot(epoch_count, test_loss_values, label="Test LOSS")
plt.title("Training and Testing LOSS Curves")
plt.ylabel("LOSS")
plt.xlabel("Epochs")
plt.legend()`,
        output: 
`<matplotlib.legend.Legend at 0x7c1760453af0>`
      },
    ]
  },
  { id: 9,
    name: "4.2_load_a_model",
    code: [
      { id: 28,
        input: 
`# Saving our PyTorch model
from pathlib import Path

# 1. Create models directory
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Create model save path
MODEL_NAME = "01_aStraightLine_model_0.pth" # .pth or .pt
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the model state dict
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_0.state_dict(),
           f=MODEL_SAVE_PATH)`,
        output: 
`Saving model to: models/01_aStraightLine_model_0.pth`
      },
      { id: 29,
        input: 
`!ls -1 models`,
        output: 
`01_aStraightLine_model_0.pth`
      },
      { id: 30,
        input: 
`model_0.state_dict()`,
        output: 
`OrderedDict([('weight', tensor([0.5632])), ('bias', tensor([0.1151]))])`
      },
    ]
  },
  { id: 10,
    name: "4.1_save_a_model",
    code: [
      { id: 31,
        input: 
`# Instantiate a new model (initial with random weight)
loaded_model_0 = LinearRegressionModel()

# Load the state_dict of saved model (updating to the trained weight)
loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))`,
        output: 
`<ipython-input-125-22e0e7e0464f>:5: FutureWarning: You are using \`torch.load\` with \`weights_only=False\` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for \`weights_only\` will be flipped to \`True\`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via \`torch.serialization.add_safe_globals\`. We recommend you start setting \`weights_only=True\` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
<All keys matched successfully>`,
      },
      { id: 32,
        input: 
`loaded_model_0.state_dict()`,
        output: 
`OrderedDict([('weight', tensor([0.5632])), ('bias', tensor([0.1151]))])`
      },
      { id: 33,
        input: 
`# Make predictions with loaded model
loaded_model_0.eval()
with torch.inference_mode():
  loaded_model_preds = loaded_model_0(X_test)

loaded_model_preds`,
        output: 
`tensor([[0.5656],
        [0.5769],
        [0.5881],
        [0.5994],
        [0.6106],
        [0.6219],
        [0.6332],
        [0.6444],
        [0.6557],
        [0.6670]])`
      },
      { id: 34,
        input: 
`# Make predictions with model_0
model_0.eval()
with torch.inference_mode():
  y_preds = model_0(X_test)
y_preds`,
        output: 
`tensor([[0.5656],
        [0.5769],
        [0.5881],
        [0.5994],
        [0.6106],
        [0.6219],
        [0.6332],
        [0.6444],
        [0.6557],
        [0.6670]])`
      },
      { id: 35,
        input: 
`# Compare the predictions of loaded model and original model
y_preds == loaded_model_preds`,
        output: 
`tensor([[True],
        [True],
        [True],
        [True],
        [True],
        [True],
        [True],
        [True],
        [True],
        [True]])`
      },
    ]
  }
]