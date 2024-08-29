export default [
  { id: 1, 
    name: "TENSORs",
    code: [
      { id: 1, 
        input: 
`import torch
torch.__version__`,
        output: 
`'1.13.1+cu116'`,
      }, 
      { id: 2, 
        input: 
`# Scalar
scalar = torch.tensor(7)
scalar`,
        output: 
`tensor(7)`,
      }, 
      { id: 3, 
        input: 
`# Vector
vector = torch.tensor([7, 7])
vector`,
        output: 
`tensor([7, 7])`,
      }, 
      { id: 4, 
        input: 
`# Matrix
Matrix = torch.tensor([[7, 8],
                       [9, 10]])
Matrix`,
        output: 
`tensor([[ 7,  8],
        [ 9, 10]])`,
      }, 
      { id: 5, 
        input: 
`# TENSOR
TENSOR = torch.tensor([[[1, 2, 3],
                        [3, 6, 9],
                        [2, 4, 5]]])
TENSOR`,
        output: 
`tensor([[[1, 2, 3],
          [3, 6, 9],
          [2, 4, 5]]])`,
      }, 
    ]
  }, 
  { id: 2,
    name: "create_tensor_random", 
    code: [
      { id: 6, 
        input: 
`# Create a random tensor with shape (3, 4)
# rand(): uniform distribution
random_tensor = torch.rand(3, 4) 
# random_tensor = torch.rand(size=(3, 4))`,
        output: 
`tensor([[0.6706, 0.0631, 0.7080],
        [0.0038, 0.8061, 0.9664],
        [0.0742, 0.1530, 0.9702]])`,
      }, 
      { id: 7, 
        input: 
`# Create another random tensor
# randn(): normal distribution
random_tensor = torch.randn(3, 4)`,
        output: 
`tensor([[-0.9213, -0.5282, -0.7235],
        [-0.0147,  0.1065, -1.1796],
        [-0.0263,  1.3543, -0.1183]])`,
      }, 
    ]
  },
  { id: 3,
    name: "create_tensor_zeros",
    code: [      
      { id: 8, 
        input: 
`# Create a tensor of all zeros
zeros = torch.zeros(size=(3, 3))`,
        output: 
`tensor([[0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]])`,
      }, 
      { id: 9, 
        input: 
`# Create a tensor of all ones
ones = torch.ones(size=(3, 4))`,
        output: 
`tensor([[1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.]])`,
      }, 
    ]
  },
  { id: 4,
    name: "create_tensor_range",
    code: [
      { id: 10, 
        input: 
`# Use torch.arange()
range_one = torch.arange(0, 10)
range_one`,
        output: 
`tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])`,
      }, 
      { id: 11, 
        input: 
`range_two = torch.arange(0, 10, 2)
# range_two = torch.arange(start=0, end=10, step=2)
range_two`,
        output: 
`tensor([0, 2, 4, 6, 8])`,
      }, 
      { id: 12, 
        input: 
`range_three = torch.linspace(0, 10, 3)
# range_three = torch.linspace(start=0, end=10, steps=3)
range_three`,
        output: 
`tensor([ 0.,  5., 10.])`,
      }, 
    ]
  },
  { id: 5,
    name: "create_tensor_likes",
    code: [
      { id: 13, 
        input: 
`# Create tensors like
zeros_like = torch.zeros_like(input=range_two)
zeros_like`,
        output: 
`tensor([0, 0, 0, 0, 0])`,
      }, 
      { id: 14, 
        input: 
`# Create tensors like
ones_like = torch.ones_like(input=range_two)
ones_like`,
        output: 
`tensor([1, 1, 1, 1, 1])`,
      }, 
    ]
  }, 
  { id: 6, 
    name: 'tensors_attributes',
    code: [
      { id: 15, 
        input: 
`# Create a tensor
a_tensor = torch.rand(3, 4)

# Find out details about some tensor
print(a_tensor)
print(f"Datatype of tensor: {a_tensor.dtype}")
print(f"Shape of tensor: {a_tensor.shape}") # or a_tensor.size()
print(f"Device tensor is on: {a_tensor.device}")`,
        output: 
`tensor([[0.2167, 0.8048, 0.1827, 0.1025],
        [0.6558, 0.4133, 0.8408, 0.6162],
        [0.5921, 0.1131, 0.5611, 0.3981]])
Datatype of tensor: torch.float32
Shape of tensor: torch.Size([3, 4])
Device tensor is on: cpu`,
      },
      { id: 16, 
        input: 
`# Default datatype for tensors is float32
float_32_tensor = torch.tensor([2.0, 4.0, 8.0], 
                               dtype=None, # dtype: defaults to None, which is torch.float32 or whatever datatype is passed
                               device=None, # device: defaults to None, which uses the default tensor type
                               requires_grad=False) # requires_grad: if True, operations performed on the tensor are recorded

# whether or not to track the gradients with this tensor operations
float_32_tensor, float_32_tensor.shape, float_32_tensor.dtype, float_32_tensor.device`,
        output: 
`(tensor([2., 4., 8.]), torch.Size([3]), torch.float32, device(type='cpu'))`,
      },
      { id: 17, 
        input: 
`float_16_tensor = float_32_tensor.type(torch.float16)
float_16_tensor`,
        output: 
`tensor([2., 4., 8.], dtype=torch.float16)`,
      },
      { id: 18, 
        input: 
`tensor_1 = float_32_tensor * float_16_tensor
tensor_1, tensor_1.dtype`,
        output: 
`(tensor([ 4., 16., 64.]), torch.float32)`,
      },
      { id: 19, 
        input: 
`int_32_tensor = torch.tensor([2, 4, 8], dtype=torch.long) 
# torch.LongTensor is a CPU tensor
int_32_tensor, int_32_tensor.dtype`,
        output: 
`(tensor([2, 4, 8]), torch.int64)`,
      },
      { id: 20, 
        input: 
`tensor_2 = float_32_tensor * int_32_tensor, 
tensor_2, (float_32_tensor * int_32_tensor).dtype`,
        output: 
`((tensor([ 4., 16., 64.]),), torch.float32)`,
      },
    ]
  },
  
  { id: 7, 
    name: 'tensor_operation_addsub',
    code: [
      { id: 21, 
        input: 
`# Create a tensor and add a number to it
tensor = torch.tensor([1, 2, 3])
tensor + 10`,
        output: 
`tensor([11, 12, 13])`,
      },
      { id: 22, 
        input: 
`# Tensor don't change unless reassigned
tensor`,
        output: 
`tensor([1, 2, 3])`,
      },
      { id: 23, 
        input: 
`torch.add(tensor, 10) # More common to use + instead torch.add()`,
        output: 
`tensor([11, 12, 13])`,
      },
      { id: 24, 
        input: 
`# Add 2 tensors with the same shape
tensor + tensor`,
        output: 
`tensor([2, 4, 6])`,
      },
      { id: 25, 
        input: 
`torch.add(tensor, tensor)`,
        output: 
`tensor([2, 4, 6])`,
      },
      { id: 26, 
        input: 
`# Subtract by a number
tensor - 10`,
        output: 
`tensor([-9, -8, -7])`,
      },
      { id: 27, 
        input: 
`# Substract with sub()
torch.sub(tensor - 10)`,
        output: 
`tensor([-9, -8, -7])`,
      },
      { id: 28, 
        input: 
`tensor`,
        output: 
`tensor([1, 2, 3])`,
      },
      { id: 29, 
        input: 
`# Subtract by a tensor with the same shape
tensor - tensor`,
        output: 
`tensor([0, 0, 0])`,
      },
      { id: 30, 
        input: 
`# Subtract with sub()
torch.sub(tensor, tensor)`,
        output: 
`tensor([0, 0, 0])`,
      },
    ]
  },
  { id: 8, 
    name: 'tensor_operation_divmul',
    code: [
      { id: 31, 
        input: 
`# Divide it by a number
tensor / 10`,
        output: 
`tensor([1.1000, 1.2000, 1.3000])`,
      },
      { id: 32, 
        input: 
`# Can also use torch functions
torch.div(tensor, 10) # build-in function`,
        output: 
`tensor([1.1000, 1.2000, 1.3000])`,
      },
      { id: 33, 
        input: 
`# Divide it by a tensor(same shape)
tensor / tensor`,
        output: 
`tensor([1., 1., 1.])`,
      },
      { id: 34, 
        input: 
`# Divide with div()
torch.div(tensor, tensor)`,
        output: 
`tensor([1., 1., 1.])`,
      },
      { id: 35, 
        input: 
`# Multiply it by 10
tensor * 10`,
        output: 
`tensor([10, 20, 30])`,
      },
      { id: 36, 
        input: 
`# Can also use torch functions
torch.mul(tensor, 10)`,
        output: 
`tensor([10, 20, 30])`,
      },
      { id: 37, 
        input: 
`# Element-wise multiplication 
tensor * tensor`,
        output: 
`tensor([1, 4, 9])`,
      },
      { id: 38, 
        input: 
`# Can also use torch functions
torch.mul(tensor, tensor)`,
        output: 
`tensor([1, 4, 9])`,
      },
    ]
  },
  { id: 9, 
    name: 'tensor_operation_matmul',
    code: [
      { id: 39, 
        input: 
`tensor`,
        output: 
`tensor([1, 2, 3])`,
      },
      { id: 40, 
        input: 
`# Matrix multiplication
torch.matmul(tensor, tensor)`,
        output: 
`tensor(14)`,
      },
      { id: 41, 
        input: 
`# Can also use the "@" symbol, but not recommend
tensor @ tensor`,
        output: 
`tensor(14)`,
      },
      { id: 42, 
        input: 
`%%time
# Matrix multiplication by hand
# (avoid doing operations with for loops at all cost, they are computationally expensive)
value = 0
for i in range(len(tensor)):
  value += tensor[i] * tensor[i]
value`,
        output: 
`CPU times: user 157 µs, sys: 28 µs, total: 185 µs
Wall time: 193 µs
tensor(14)`,
      },
      { id: 43, 
        input: 
`%%time
torch.matmul(tensor, tensor)`,
        output: 
`CPU times: user 29 µs, sys: 5 µs, total: 34 µs
Wall time: 38.1 µs
tensor(14)`,
      },
    ]
  },
  { id: 10, 
    name: 'tensor_manipulate_aggregate',
    code: [
      { id: 44, 
        input: 
`# Create a tensor
x = torch.arange(0, 100, 10)
x, x.dtype`,
        output: 
`(tensor([ 0, 10, 20, 30, 40, 50, 60, 70, 80, 90]), torch.int64)`,
      },
      { id: 45, 
        input: 
`print(f"Minimum: {x.min()}")
print(f"Maximum: {x.max()}\n")

# print(f"Mean: {x.mean()}") # error’s here —>
# won't work without float datatype
print(f"Mean: {x.type(torch.float32).mean()}”)
print(f"Sum: {x.sum()}")`,
        output: 
`Minimum: 0
Maximum: 90

Mean: 45.0
Sum: 450`,
      },
      { id: 46, 
        input: 
`torch.max(x), torch.min(x), torch.mean(x.type(torch.float32)), torch.sum(x)`,
        output: 
`(tensor(90), tensor(0), tensor(45.), tensor(450))`,
      },
      { id: 47, 
        input: 
`# Create a tensor
tensor = torch.arange(10, 100, 20)
print(f"Tensor: {tensor}")

# Return index of max/min values
print(f"Index where max value occurs: {tensor.argmax()}")
print(f"Index where min value occurs: {tensor.argmin()}")`,
        output: 
`Tensor: tensor([10, 30, 50, 70, 90])
Index where max value occurs: 4
Index where min value occurs: 0`,
      },
    ]
  },
  { id: 11, 
    name: 'tensor_manipulate_reshape',
    code: [
      { id: 48, 
        input: 
`# Create a tensor and check its datatype
tensor = torch.arange(10., 100., 20.)
tensor.dtype`,
        output: 
`tensor([10., 30., 50., 70., 90.], dtype=torch.float16)`,
      },
      { id: 49, 
        input: 
`# float32 -> float16
tensor_float16 = tensor.type(torch.float16)
tensor_float16`,
        output: 
`tensor([10., 30., 50., 70., 90.], dtype=torch.float16)`,
      },
      { id: 50, 
        input: 
`# Create a int8 tensor
tensor_int8 = tensor.type(torch.int8)
tensor_int8`,
        output: 
`tensor([10, 30, 50, 70, 90], dtype=torch.int8)`,
      },
      { id: 51, 
        input: 
`# Create a tensor
import torch
x = torch.arange(1., 8.)
x, x.shape`,
        output: 
`(tensor([1., 2., 3., 4., 5., 6., 7.]), torch.Size([7]))`,
      },
      { id: 52, 
        input: 
`# reshape(): Add an extra dimension
x_reshaped = x.reshape(1, 7)
# x_reshaped = torch.reshape(x, (1, 7))
x_reshaped, x_reshaped.shape`,
        output: 
`(tensor([[1., 2., 3., 4., 5., 6., 7.]]), torch.Size([1, 7]))`,
      },
      { id: 53, 
        input: 
`x`,
        output: 
`tensor([1., 2., 3., 4., 5., 6., 7.])`,
      },
      { id: 54, 
        input: 
`# Change view
z = x.view(1, 7)
print(f"Original tensor: {x, x.shape}")
print(f"Change the view: {z, z.shape}")`,
        output: 
`Original tensor: (tensor([1., 2., 3., 4., 5., 6., 7.]), torch.Size([7]))
Change the view: (tensor([[1., 2., 3., 4., 5., 6., 7.]]), torch.Size([1, 7]))`,
      },
      { id: 55, 
        input: 
`# Change z -> x changed
# (cus a view of tensor shares the same memory as the original input)
z[:, 0] = 5
z, x`,
        output: 
`(tensor([[5., 2., 3., 4., 5., 6., 7.]]), tensor([5., 2., 3., 4., 5., 6., 7.]))`,
      },
      { id: 56, 
        input: 
``,
        output: 
``,
      },
      { id: 57, 
        input: 
``,
        output: 
``,
      },
      { id: 58, 
        input: 
``,
        output: 
``,
      },
    ]
  },
  { id: 12, 
    name: 'tensor_manipulate_catstack',
    code: [
      { id: 59, 
        input: 
`# Create a tensor
x = torch.arange(0, 100, 10)
x, x.dtype`,
        output: 
`(tensor([ 0, 10, 20, 30, 40, 50, 60, 70, 80, 90]), torch.int64)`,
      },
      { id: 60, 
        input: 
`# Stack tensors on top of each other
x_stacked = torch.stack([x, x+1, x+2], dim=0)
x_stacked`,
        output: 
`tensor([[5., 2., 3., 4., 5., 6., 7.],
        [6., 3., 4., 5., 6., 7., 8.],
        [7., 4., 5., 6., 7., 8., 9.]])`,
      },
      { id: 61, 
        input: 
`x_stacked = torch.stack([x, x+1, x+2], dim=1)
x_stacked`,
        output: 
`tensor([[5., 6., 7.],
        [2., 3., 4.],
        [3., 4., 5.],
        [4., 5., 6.],
        [5., 6., 7.],
        [6., 7., 8.],
        [7., 8., 9.]])`,
      },
      { id: 62, 
        input: 
`# torch.permute() - rearranges the dimensions of a target tensor in a specified order
x_original = torch.rand(size=(224, 224, 3)) #[height, width, color_channels]

# Permute the original tensor to rearrange the axis/dim order
x_permuted = x_original.permute(2, 0, 1) # Shifts axis 0->1, 1->2, 2->0

print(f"Previous shape: {x_original.shape}")
print(f"New shape: {x_permuted.shape}") # [color_channels, height, width]`,
        output: 
`Previous shape: torch.Size([224, 224, 3])
New shape: torch.Size([3, 224, 224])`,
      },
    ]
  },
]