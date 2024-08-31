export default [
  { id: 1, 
    name: "TENSORs",
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
`# Scalar
scalar = torch.tensor(0)
scalar`,
        output: 
`tensor(0)`,
      }, 
      { id: 3, 
        input: 
`# Vector
vector = torch.tensor([0, 1])
vector`,
        output: 
`tensor([0, 1])`,
      }, 
      { id: 4, 
        input: 
`# MATRIX
MATRIX = torch.tensor([[0, 1],
                       [1, 2]])
MATRIX`,
        output: 
`tensor([[ 0, 1],
        [ 1, 2]])`,
      }, 
      { id: 5, 
        input: 
`# TENSOR
TENSOR = torch.tensor([[[0, 1, 2],
                        [1, 2, 3],
                        [2, 3, 4]]])
TENSOR`,
        output: 
`tensor([[[0, 1, 2],
         [1, 2, 3],
         [2, 3, 4]]])`,
      }, 
    ]
  }, 
  { id: 2,
    name: "create_tensor_random", 
    code: [
      { id: 6, 
        input: 
`# Create a random tensor with shape (3, 3)
# rand(): uniform distribution
tensor_rand = torch.rand(3, 3) 
# tensor_rand = torch.rand(size=(3, 3))`,
        output: 
`tensor([[0.0088, 0.0259, 0.8963],
        [0.1246, 0.0526, 0.0891],
        [0.4923, 0.1178, 0.3876]])`,
      }, 
      { id: 7, 
        input: 
`# Create another random tensor with 
# randn(): normal distribution
tensor_randn = torch.randn(3, 3)`,
        output: 
`tensor([[0.1229, 0.6476, 0.1100],
        [0.7567, 0.7385, 0.2270],
        [0.3819, 0.7599, 0.8160]])`,
      }, 
    ]
  },
  { id: 3,
    name: "create_tensor_zeros",
    code: [      
      { id: 8, 
        input: 
`# Create a tensor full of zeros
tensor_zeros = torch.zeros(size=(3, 3))`,
        output: 
`tensor([[0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]])`,
      }, 
      { id: 9, 
        input: 
`# Create a tensor full of ones
tensor_ones = torch.ones(size=(3, 3))`,
        output: 
`tensor([[1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.]])`,
      }, 
    ]
  },
  { id: 4,
    name: "create_tensor_range",
    code: [
      { id: 10, 
        input: 
`# Create a range of tensor
tensor_arange = torch.arange(0, 5)
tensor_arange`,
        output: 
`tensor([0, 1, 2, 3, 4])`,
      }, 
      { id: 11, 
        input: 
`tensor_arange_two = torch.arange(0, 10, 2)
# tensor_arange_two = torch.arange(start=0, end=10, step=2)
tensor_arange_two`,
        output: 
`tensor([0, 2, 4, 6, 8])`,
      }, 
    ]
  },
  { id: 5,
    name: "create_tensor_likes",
    code: [
      { id: 12, 
        input: 
`# Create tensor like - zeros like
zeros_like = torch.zeros_like(input=tensor_arange_two)
zeros_like`,
        output: 
`tensor([0, 0, 0, 0, 0])`,
      }, 
      { id: 13, 
        input: 
`# Create tensor like - ones like
ones_like = torch.ones_like(input=tensor_arange_two)
ones_like`,
        output: 
`tensor([1, 1, 1, 1, 1])`,
      }, 
    ]
  }, 
  { id: 6, 
    name: "tensors_attributes",
    code: [
      { id: 14, 
        input: 
`import torch

# Create a tensor with shape 3*3
tensor_33 = torch.randn(3, 3)

# Let's check the attributes of this tensor
print(f"{tensor_33}\\n")
print(f"Shape of tensor: {tensor_33.shape}") # or tensor_33.size()
print(f"Dim Number of tensor: {tensor_33.ndim}")
print(f"Data type of tensor: {tensor_33.dtype}")
print(f"Device tensor is on: {tensor_33.device}")`,
        output: 
`tensor([[-0.1240,  1.7736, -1.6758],
        [-0.8741,  0.7059, -0.4879],
        [-1.5880,  0.8205,  0.4376]])

Shape of tensor: torch.Size([3, 3])
Dim Number of tensor: 2
Data type of tensor: torch.float32
Device tensor is on: cpu`,
      },
    ]
  },
  
  { id: 7, 
    name: "tensor_operation_addsub",
    code: [
      { id: 15, 
        input: 
`import torch

# Create a tensor and add a number to it
tensor_add = torch.tensor([1, 2, 3])
tensor_add + 10`,
        output: 
`tensor([11, 12, 13])`,
      },
      { id: 16, 
        input: 
`tensor_add # tensor_add don't change unless reassign it`,
        output: 
`tensor([1, 2, 3])`,
      },
      { id: 17, 
        input: 
`# Here we can use add(), 
# but we prefere for '+', cus it's more common and readable
# tensor_add.add(10)
torch.add(tensor_add, 10)`,
        output: 
`tensor([11, 12, 13])`,
      },
      { id: 18, 
        input: 
`# Add 2 tensors with the same shape
tensor_add + tensor_add`,
        output: 
`tensor([2, 4, 6])`,
      },
      { id: 19, 
        input: 
`torch.add(tensor_add, tensor_add)`,
        output: 
`tensor([2, 4, 6])`,
      },
      { id: 20, 
        input: 
`# Create a tensor then subtract a single number
tensor_sub = torch.tensor([1, 2, 3])
tensor_sub - 10`,
        output: 
`tensor([-9, -8, -7])`,
      },
      { id: 21, 
        input: 
`torch.sub(tensor_sub, 10)`,
        output: 
`tensor([-9, -8, -7])`,
      },
      { id: 22, 
        input: 
`# Subtract a tensor with the same shape
tensor_sub - tensor_sub`,
        output: 
`tensor([0, 0, 0])`,
      },
      { id: 23, 
        input: 
`torch.sub(tensor_sub, tensor_sub)`,
        output: 
`tensor([0, 0, 0])`,
      },
      { id: 24, 
        input: 
`# Create a tensor then divide it by a single number
tensor_div = torch.tensor([1, 2, 3])
tensor_div / 10`,
        output: 
`tensor([1.1000, 1.2000, 1.3000])`,
      },
      { id: 25, 
        input: 
`# Can also use torch function
torch.div(tensor_div, 10) # build-in function`,
        output: 
`tensor([1.1000, 1.2000, 1.3000])`,
      },
      { id: 26, 
        input: 
`# Divide it by a tensor (same shape)
tensor_div / tensor_div`,
        output: 
`tensor([1., 1., 1.])`,
      },
      { id: 27, 
        input: 
`# Divide with div()
torch.div(tensor_div, tensor_div)`,
        output: 
`tensor([1., 1., 1.])`,
      },
      { id: 28, 
        input: 
`# Create a tensor then multiply it by a single number
tensor_mul = torch.tensor([1, 2, 3])
tensor_mul * 10`,
        output: 
`tensor([10, 20, 30])`,
      },
      { id: 29, 
        input: 
`# Can also use torch functions
torch.mul(tensor_mul, 10)`,
        output: 
`tensor([10, 20, 30])`,
      },
      { id: 30, 
        input: 
`# Element-wise multiplication 
tensor_mul * tensor_mul`,
        output: 
`tensor([1, 4, 9])`,
      },
      { id: 31, 
        input: 
`# Can also use torch functions
torch.mul(tensor_mul, tensor_mul)`,
        output: 
`tensor([1, 4, 9])`,
      },
    ]
  },
  { id: 8, 
    name: "tensor_operation_matmul",
    code: [
      { id: 32, 
        input: 
`import torch

tensor_matmul = torch.tensor([1, 2, 3]) # vector`,
        output: 
`tensor([1, 2, 3])`,
      },
      { id: 33, 
        input: 
`# Matrix Multiplication
tensor_matmul.matmul(tensor_matmul) # dot product`,
        output: 
`tensor(14)`,
      },
      { id: 34, 
        input: 
`# Prefer '@', clear and concise
tensor_matmul @ tensor_matmul`,
        output: 
`tensor(14)`,
      },
      { id: 35, 
        input: 
`%%time
# Matrix multiplication by hand
# (avoid doing operations with for loops at all cost, they are computationally expensive)
value = 0
for i in range(len(tensor_matmul)):
  value += tensor_matmul[i] * tensor_matmul[i]
value`,
        output: 
`CPU times: user 995 µs, sys: 0 ns, total: 995 µs
Wall time: 2.22 ms
tensor(14)`,
      },
      { id: 36, 
        input: 
`%%time
torch.matmul(tensor_matmul, tensor_matmul)`,
        output: 
`CPU times: user 138 µs, sys: 0 ns, total: 138 µs
Wall time: 142 µs
tensor(14)`,
      },
      { id: 37, 
        input: 
`# Create 2 matrices, and matrix multiplication
tensor_matmul_1 = torch.tensor([[1, 2, 0],
                                [0, 1, 0]])
tensor_matmul_2 = torch.tensor([[2, 1],
                                [0, 1],
                                [6, 8]])

tensor_3 = tensor_matmul_1 @ tensor_matmul_2
tensor_3`,
        output: 
`tensor([[2, 3],
        [0, 1]])`,
      },
      { id: 38, 
        input: 
`# Check the tensors' shape
print(f"Shape of tensor_matmul_1: {tensor_matmul_1.shape}")
print(f"Shape of tensor_matmul_2: {tensor_matmul_2.shape}\\n")
print(f"Shape of tensor_3: {tensor_3.shape}")`,
        output: 
`Shape of tensor_matmul_1: torch.Size([2, 3])
Shape of tensor_matmul_2: torch.Size([3, 2])

Shape of tensor_3: torch.Size([2, 2])`,
      },
    ]
  },
  { id: 9, 
    name: "tensor_operation_aggregate",
    code: [
      { id: 39, 
        input: 
`# Create a tensor
x = torch.arange(0, 50, 10)
x, x.dtype`,
        output: 
`(tensor([ 0, 10, 20, 30, 40]), torch.int64)`,
      },
      { id: 40, 
        input: 
`print(f"Minimum: {x.min()}")
print(f"Maximum: {x.max()}")
print(f"Sum: {x.sum()}")`,
        output: 
`Minimum: 0
Maximum: 40
Sum: 100`,
      },
      { id: 41, 
        input: 
`print(f"Mean: {x.mean()}") # error's here —>`,
        output: 
`---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
<ipython-input-14-a6e7e2ee83ab> in <cell line: 1>()
----> 1 print(f"Mean: {x.mean()}") # error's here —>
      2 # mean(): won't work without float datatype
      3 # print(f"Mean: {x.type(torch.float32).mean()}")

RuntimeError: mean(): could not infer output dtype. Input dtype must be either a floating point or complex dtype. Got: Long`,
      },
      { id: 42, 
        input: 
`# mean(): won't work without float datatype
print(f"Mean: {x.type(torch.float32).mean()}")`,
        output: 
`Mean: 20.0`,
      },
      { id: 43, 
        input: 
`torch.max(x), torch.min(x), torch.mean(x.type(torch.float32)), torch.sum(x)`,
        output: 
`(tensor(40), tensor(0), tensor(20.), tensor(100))`,
      },
      { id: 44, 
        input: 
`print(f"Tensor: {x}")

# Return index of max/min values
print(f"Index of the maximum value: {x.argmax()}")
print(f"Index of the minimum value: {x.argmin()}")`,
        output: 
`Tensor: tensor([ 0, 10, 20, 30, 40])
Index of the maximum value: 4
Index of the minimum value: 0`,
      },
    ]
  },
  { id: 10, 
    name: "tensor_manipulate_reshape",
    code: [
      { id: 45, 
        input: 
`import torch

# Create a tensor and check its data type
x = torch.tensor([0., 10., 20., 30., 40.])
x, x.dtype`,
        output: 
`(tensor([ 0., 10., 20., 30., 40.]), torch.float32)`,
      },
      { id: 46, 
        input: 
`# float32 -> float16
tensor_f16 = x.type(torch.float16)
tensor_f16`,
        output: 
`tensor([ 0., 10., 20., 30., 40.], dtype=torch.float16)`,
      },
      { id: 47, 
        input: 
`# Create a certain data type tensor
tensor_int8 = torch.tensor([0, 1, 2, 3], dtype=torch.int8)
tensor_int8`,
        output: 
`tensor([0, 1, 2, 3], dtype=torch.int8)`,
      },
      { id: 48, 
        input: 
`import torch

# Create a tensor and check its data type
x = torch.tensor([0., 10., 20., 30., 40.])

# Check the tensor shape
x.shape`,
        output: 
`torch.Size([5])`,
      },
      { id: 49, 
        input: 
`# add an extra dim with reshape()
x_reshape_1 = x.reshape(5, 1) # add an inner dim
x_reshape_2 = x.reshape(1, 5) # add an outer dim

print(x_reshape_1)
print(x_reshape_2)`,
        output: 
`tensor([[9.],
        [2.],
        [4.],
        [6.],
        [8.]])
tensor([[9., 2., 4., 6., 8.]])`,
      },
      { id: 50, 
        input: 
`print(f"Original tensor:\\n {x}\\n")

x_reshape_2[:, 0] = 9

# change x_reshape_2, x changed
# (reason: share the same memory)
print(f"Change the reshaped tensor:\\n {x_reshape_2}\\n")
print(f"Original tensor changed:\\n {x}")`,
        output: 
`Original tensor:
 tensor([ 0., 10., 20., 30., 40.])

Change the reshaped tensor:
 tensor([[9., 2., 4., 6., 8.]])

Original tensor changed:
 tensor([ 0., 10., 20., 30., 40.])`,
      },
      { id: 51, 
        input: 
`# change tensor shape with view()
x_view_3 = x.view(5, 1)
x_view_4 = x.view(1, 5)

print(f"Change the view (inner dim):\\n {x_view_3}")
print(f"Change the view (outer dim):\\n {x_view_4}\\n")
print(f"Original tensor:\\n {x}")`,
        output: 
`Change the view (inner dim):
 tensor([[ 0.],
        [10.],
        [20.],
        [30.],
        [40.]])
Change the view (outer dim):
 tensor([[ 0., 10., 20., 30., 40.]])

Original tensor:
 tensor([ 0., 10., 20., 30., 40.])`,
      },
      { id: 52, 
        input: 
`# squeeze(): remove all single dimensions from a target tensor
print(f"Previous tensor:\\n {x_view_4, x_view_4.shape}]\\n")

# Remove extra dimensions from x_reshaped
x_view_4_squeezed = x_view_4.squeeze()

print(f"Squeezed tensor:\\n {x_view_4_squeezed, x_view_4_squeezed.shape}")`,
        output: 
`Previous tensor:
 (tensor([[ 0., 10., 20., 30., 40.]]), torch.Size([1, 5]))

Squeezed tensor:
 (tensor([ 0., 10., 20., 30., 40.]), torch.Size([5]))`,
      },
      { id: 53, 
        input: 
`# unsqueeze(): adds a single dimension to a target tensor at a specific dim
print(f"Previous tensor:\\n {x_view_4_squeezed, x_view_4_squeezed.shape}\\n")

# Add an extra dimension with unsqueeze()
x_view_4_unsqueezed = x_view_4_squeezed.unsqueeze(dim=0)

print(f"Squeezed tensor:\\n {x_view_4_unsqueezed, x_view_4_unsqueezed.shape}")`,
        output: 
`Previous tensor:
 (tensor([ 0., 10., 20., 30., 40.]), torch.Size([5]))

Squeezed tensor:
 (tensor([[ 0., 10., 20., 30., 40.]]), torch.Size([1, 5]))`,
      },
      { id: 54, 
        input: 
`# Create a tensor with a specific shape
img = torch.rand(size=[128, 128, 3])

print(f"Original tensor shape: {img.shape}")

# permute(): rearrange the axis order
p = img.permute(2, 0, 1)
print(f"Permuted tensor shape: {p.shape}")`,
        output: 
`Original tensor shape: torch.Size([128, 128, 3])
Permuted tensor shape: torch.Size([3, 128, 128])`,
      },
    ]
  },

  { id: 11, 
    name: "tensor_manipulate_transpose",
    code: [
      { id: 59, 
        input: 
`import torch

# Create 2 tensors and matrix multiplicate them
A = torch.tensor([[1., 2., 3.], 
                [4., 5., 6.]])
B = torch.tensor([[2., 3., 4.],
                [5., 6., 7.]])

torch.matmul(A, B) # error's here`,
        output: 
`---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
<ipython-input-63-d1c1c19eddeb> in <cell line: 9>()
        7                   [5., 6., 7.]])
        8 
----> 9 torch.matmul(A, B) # error's here

RuntimeError: mat1 and mat2 shapes cannot be multiplied (2x3 and 2x3)`,
      },
      { id: 60, 
        input: 
`# Their inner shapes are not matched
print(f"Shape of A: {A.shape}")
print(f"Shape of B: {B.shape}\\n")

# Transpose B to match the inner dim
print(f"Original B shape: {B.shape}")
print(f"Transposed B shape: {B.T.shape}\\n")

# A @ B.T
output = torch.matmul(A, B.T)

print(f"Matrix Multiplicate :\\n {A.shape} * {B.T.shape} <- inner dim match")
print(f"Output:")
print(f"Output shape: {output.shape}")`,
        output: 
`Shape of A: torch.Size([2, 3])
Shape of B: torch.Size([2, 3])

Original B shape: torch.Size([2, 3])
Transposed B shape: torch.Size([3, 2])

Matrix Multiplicate :
torch.Size([2, 3]) * torch.Size([3, 2]) <- inner dim match
Output:
Output shape: torch.Size([2, 2])`,
      },
      { id: 61, 
        input: 
`# transpose(input, dim0, dim1)
B_transposed = torch.transpose(B, 0, 1)

print(f"Original B shape: {B.shape}")
print(f"Transposed B shape: {B_transposed.shape}\\n")

result = A @ B_transposed
print(f"Matrix Multiplicate :\\n {A.shape} * {B_transposed.shape} <- inner dim match")
print(f"Output:")
print(f"Output shape: {result.shape}")`,
        output: 
`Original B shape: torch.Size([2, 3])
Transposed B shape: torch.Size([3, 2])

Matrix Multiplicate :
torch.Size([2, 3]) * torch.Size([3, 2]) <- inner dim match
Output:
Output shape: torch.Size([2, 2])`,
      },
    ]
  },
  { id: 12, 
    name: "tensor_numpy",
    code: [
      { id: 82, 
        input: 
`# NumPy array to tensor
import torch
import numpy as np

# Numpy Array -> Tensor
array = np.arange(1.0, 8.0)
tensor = torch.from_numpy(array)

array, tensor`, 
        output: 
`(array([1., 2., 3., 4., 5., 6., 7.]),
tensor([1., 2., 3., 4., 5., 6., 7.], dtype=torch.float64))`,
      },
      { id: 83, 
        input: 
`# Change the array, keep the tensor
array = array + 1
array, tensor`, 
        output: 
`(array([2., 3., 4., 5., 6., 7., 8.]),
tensor([1., 2., 3., 4., 5., 6., 7.], dtype=torch.float64))`,
      },
      { id: 84, 
        input: 
`array.dtype, tensor.dtype`, 
        output: 
`(dtype('float64'), torch.float64)`,
      },
      { id: 85, 
        input: 
`# Tensor -> NumPy array
tensor = torch.ones(7) 
numpy_tensor = tensor.numpy()
tensor, numpy_tensor`, 
        output: 
`(tensor([1., 1., 1., 1., 1., 1., 1.]),
array([1., 1., 1., 1., 1., 1., 1.], dtype=float32))`,
      },
      { id: 86, 
        input: 
`# Change the tensor, what happens to numpy_tensor?
tensor = tensor + 1
tensor, numpy_tensor`, 
        output: 
`(tensor([2., 2., 2., 2., 2., 2., 2.]),
array([1., 1., 1., 1., 1., 1., 1.], dtype=float32))`,
      },
    ]
  }, 
  { id: 13, 
    name: "tensor_manipulate_catstack",
    code: [
      { id: 55, 
        input: 
`import torch

# Create 2 tensors
A = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
B = torch.tensor([[2., 3., 4.], [5., 6., 7.]])

# Concatenate A and B on the 1st dim
A_cat_B = torch.cat([A, B], dim=0)
print(f"Concatenated tensor:\\n {A_cat_B}\\n")

# Concatenate A and B on the 2nd dim
A_cat_B_2 = torch.cat([A, B], dim=1)
print(f"Concatenated tensor:\\n {A_cat_B_2}")`,
        output: 
`Concatenated tensor:
 tensor([[1., 2., 3.],
        [4., 5., 6.],
        [2., 3., 4.],
        [5., 6., 7.]])

Concatenated tensor:
 tensor([[1., 2., 3., 2., 3., 4.],
        [4., 5., 6., 5., 6., 7.]])`,
      },
      { id: 56, 
        input: 
`import torch

# create a tensor 
A = torch.tensor([1., 2., 3., 4., 5.])
A, A.shape`,
        output: 
`(tensor([1., 2., 3., 4., 5.]), torch.Size([5]))`,
      },
      { id: 57, 
        input: 
`# Stack tensors on top of each other
A_stacked = torch.stack([A, A+1, A+2], dim=0)

# We can use vstack() to get the same result
A_vstacked = torch.vstack([A, A+1, A+2])

print(f"{A_stacked, A_stacked.shape}\\n")
print(A_vstacked, A_vstacked.shape)`,
        output: 
`(tensor([[1., 2., 3., 4., 5.],
        [2., 3., 4., 5., 6.],
        [3., 4., 5., 6., 7.]]), torch.Size([3, 5]))

tensor([[1., 2., 3., 4., 5.],
        [2., 3., 4., 5., 6.],
        [3., 4., 5., 6., 7.]]) torch.Size([3, 5])`,
      },
      { id: 58, 
        input: 
`# Stack tensors on the second dim
A_stacked = torch.stack([A, A+1], dim=1)

# vstack(): horizontally stack along the last dim
A_hstacked = torch.hstack([A, A+1])

print(f"{A_stacked, A_stacked.shape}\\n")
print(A_hstacked, A_hstacked.shape)`,
        output: 
`(tensor([[1., 2.],
        [2., 3.],
        [3., 4.],
        [4., 5.],
        [5., 6.]]), torch.Size([5, 2]))

tensor([1., 2., 3., 4., 5., 2., 3., 4., 5., 6.]) torch.Size([10])`,
      },
    ]
  },
  { id: 14, 
    name: "tensor_indexing_rowcolumn",
    code: [
      {
        id: 71, 
        input: 
`import torch

# Create a tensor
A_index = torch.arange(0, 9).reshape(1, 3, 3)
A_index, A_index.shape`,
        output: 
`(tensor([[[0, 1, 2],
          [3, 4, 5],
          [6, 7, 8]]]),
 torch.Size([1, 3, 3]))`,
      },
      {
        id: 72, 
        input: 
`# Let's index this tensor by bracket
print(f"The outer square bracket: \\n{A_index[0]}")
print(f"The middle squre bracket: {A_index[0][0]}")
print(f"The inner square bracket: {A_index[0][0][0]}")`,
        output: 
`The outer square bracket: 
tensor([[0, 1, 2],
        [3, 4, 5],
        [6, 7, 8]])
The middle squre bracket: tensor([0, 1, 2])
The inner square bracket: 0`,
      },
      {
        id: 73, 
        input: 
`# Another way to index tensor with ','
print(f"The outer square bracket: \\n{A_index[0]}")
print(f"The middle squre bracket: {A_index[0, 0]}")
print(f"The inner square bracket: {A_index[0, 0, 0]}")`,
        output: 
`The outer square bracket: 
tensor([[0, 1, 2],
        [3, 4, 5],
        [6, 7, 8]])
The middle squre bracket: tensor([0, 1, 2])
The inner square bracket: 0`,
      },
      {
        id: 74, 
        input: 
`# Index a particular number('8')
A_index[0][2][2]`,
        output: 
`tensor(8)`,
      },
      {
        id: 75, 
        input: 
`A_index[0, 2, 2]`,
        output: 
`tensor(8)`,
      },
      {
        id: 76, 
        input: 
`A_index[:, 2, 2] # ':' - means all values in current dim`,
        output: 
`tensor([8])`,
      },
      {
        id: 77, 
        input: 
`# Row/Column Indexing
A_index[:, 0]`,
        output: 
`tensor([[0, 1, 2]])`,
      },
      {
        id: 78, 
        input: 
`A_index[:, :, 1]`,
        output: 
`tensor([[1, 4, 7]])`,
      },
    ]
  },
  { id: 15, 
    name: "tensor_indexing_sliceboolean",
    code: [
      {
        id: 79,
        input: 
`# Slicing Indexing
# (extract a range of elements form a tensor)
A_index[:, :2, :2] # ':2" - means get the 1st 2 elements in current dim`,
        output: 
`tensor([[[0, 1],
         [3, 4]]])`,
      },
      {
        id: 80,
        input: 
`# (specific a step value in a tensor)
A_index[:, :2, ::2] # '::2' - means take every 2nd element`,
        output: 
`tensor([[[0, 2],
         [3, 5]]])`,
      },
      {
        id: 81,
        input: 
`# Boolean Indexing
# (with a condition to index tensor)
A_index[A_index > 3]`,
        output: 
`tensor([4, 5, 6, 7, 8])`,
      },
    ]
  },
  { id: 16, 
    name: "tensor_randomness",
    code: [
      { id: 87,
        input: 
`import torch

# Create 2 random tensors
random_tensor_A = torch.rand(3, 2)
random_tensor_B = torch.rand(3, 2)

print(f"Tensor A: \\n{random_tensor_A}")
print(f"Tensor B: \\n{random_tensor_B}\\n")
print(f"Does Tensor A equal Tensor B?")
random_tensor_A == random_tensor_B`,
        output: 
`Tensor A: 
tensor([[0.7278, 0.4802],
        [0.4283, 0.2014],
        [0.8975, 0.2648]])
Tensor B: 
tensor([[0.9012, 0.0164],
        [0.3801, 0.4905],
        [0.8460, 0.0056]])

Does Tensor A equal Tensor B?
tensor([[False, False],
        [False, False],
        [False, False]])`,
       },
    ]
  },
  { id: 17,
    name: "tensor_reproducibility",
    code: [
      { id: 88, 
        input:
`# Let's make some random but reproducible tensors
import torch

# Set random seed fro reproducibility
RANDOM_SEED = 42 # you could choose any number here

torch.manual_seed(RANDOM_SEED)
random_tensor_A = torch.rand(3, 4)

torch.manual_seed(RANDOM_SEED)
random_tensor_B = torch.rand(3, 4)

print(f"Tensor A: \\n{random_tensor_A}\\n")
print(f"Tensor B: \\n{random_tensor_B}\\n")
print(f"Does Tensor A equal Tensor B?")
random_tensor_A == random_tensor_B`,
        output:
`Tensor A: 
tensor([[0.8823, 0.9150, 0.3829, 0.9593],
        [0.3904, 0.6009, 0.2566, 0.7936],
        [0.9408, 0.1332, 0.9346, 0.5936]])

Tensor B: 
tensor([[0.8823, 0.9150, 0.3829, 0.9593],
        [0.3904, 0.6009, 0.2566, 0.7936],
        [0.9408, 0.1332, 0.9346, 0.5936]])

Does Tensor A equal Tensor B?
tensor([[True, True, True, True],
        [True, True, True, True],
        [True, True, True, True]])`,
      },
    ]
  },
  { id: 18,
    name: "tensor_gpus",
    code: [
      { id: 89,
        input: 
`!nvidia-smi`,
        output:
`Fri May  5 07:59:22 2023       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.85.12    Driver Version: 525.85.12    CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla T4            Off  | 00000000:00:04.0 Off |                    0 |
| N/A   35C    P8     9W /  70W |      0MiB / 15360MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+`,
       },
       { id: 90,
        input: 
`# check if GPU is available
import torch
torch.cuda.is_available()`,
        output:
`TRUE`,
       },
       { id: 91,
        input: 
`# Set Device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
device`,
        output:
`'cuda'`,
       },
       { id: 92,
        input: 
`# Count number of devices
torch.cuda.device_count()`,
        output:
`1`,
       },
       { id: 93,
        input: 
`# Create a tensor (default on the CPU)
tensor = torch.tensor([1, 2, 3])

# Tensor not on GPU
print(tensor, tensor.device)

# Move tensor to GPU (if available)
tensor_on_gpu = tensor.to(device)
tensor_on_gpu`,
        output:
`tensor([1, 2, 3]) cpu
tensor([1, 2, 3], device='cuda:0')`,
       },
       { id: 94,
        input: 
`# If tensor is on GPU, can't transform it to NumPy
tensor_on_gpu.numpy()`,
        output:
`-------------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-64-53175578f49e> in <cell line: 2>()
      1 # If tensor is on GPU, can't transform it to NumPy (error)
----> 2 tensor_on_gpu.numpy()

TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.`,
       },
       { id: 95,
        input: 
`# Instead, copy the tensor back to cpu
tensor_back_on_cpu = tensor_on_gpu.cpu().numpy()
tensor_back_on_cpu`,
        output:
`array([1, 2, 3])`,
       },
       { id: 96,
        input: 
`# it returns a copy of the GPU tensor in CPU memory, so the original tensor is still on GPU
tensor_on_gpu`,
        output:
`tensor([1, 2, 3], device='cuda:0')`,
       },
    ]
  }
]