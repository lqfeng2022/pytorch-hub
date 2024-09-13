export default [
  { id: 0, 
    name: "1_TENSORs",
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
vector = torch.tensor([0, 1, 2])
vector`,
        output: 
`tensor([0, 1, 2])`,
      }, 
      { id: 4, 
        input: 
`# MATRIX
MATRIX = torch.tensor([[0, 1, 2],
                       [1, 2, 3],
                       [2, 3, 4]])
MATRIX`,
        output: 
`tensor([[0, 1, 2],
        [1, 2, 3],
        [2, 3, 4]])`,
      }, 
      { id: 5, 
        input: 
`# TENSOR
TENSOR = torch.tensor([[[0, 1, 2],
                        [1, 2, 3],
                        [2, 3, 4]],
                       [[1, 2, 3],
                        [2, 3, 4],
                        [3, 4, 5]]])
TENSOR`,
        output: 
`tensor([[[0, 1, 2],
         [1, 2, 3],
         [2, 3, 4]],

        [[1, 2, 3],
         [2, 3, 4],
         [3, 4, 5]]])`,
      }, 
    ]
  }, 
  { id: 1,
    name: "2.1_tensor_create_random", 
    code: [
      { id: 6, 
        input: 
`import torch

# Create a random tensor with rand()
tensor_rand = torch.rand(3, 3) # uniform distribution
# tensor_rand = torch.rand(size=(3, 3))`,
        output: 
`tensor([[0.0088, 0.0259, 0.8963],
        [0.1246, 0.0526, 0.0891],
        [0.4923, 0.1178, 0.3876]])`,
      }, 
      { id: 7, 
        input: 
`# Create another random tensor with randn()
tensor_randn = torch.randn(3, 3) # normal distribution`,
        output: 
`tensor([[0.1229, 0.6476, 0.1100],
        [0.7567, 0.7385, 0.2270],
        [0.3819, 0.7599, 0.8160]])`,
      }, 
    ]
  },
  { id: 2,
    name: "2.2_tensor_create_zeros",
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
  { id: 3,
    name: "2.3_tensor_create_range",
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
`tensor_arange_2 = torch.arange(0, 10, 2)
# tensor_arange_2 = torch.arange(start=0, end=10, step=2)
tensor_arange_2`,
        output: 
`tensor([0, 2, 4, 6, 8])`,
      }, 
      { id: 12,
        input: 
`tensor_range = torch.range(0, 5)
tensor_range # Warning's here
# issue: torch.range() includes the end value`,
        output:
`<ipython-input-19-37d0b3f49663>:1: UserWarning: torch.range is deprecated and will be removed in a future release because its behavior is inconsistent with Python's range builtin. Instead, use torch.arange, which produces values in [start, end).
  tensor_range = torch.range(0, 5)
tensor([0., 1., 2., 3., 4., 5.])`,
      }
    ]
  },
  { id: 4,
    name: "2.4_tensor_create_likes",
    code: [
      { id: 13, 
        input: 
`x = torch.tensor([0, 1, 2, 3, 4])

# Create a zeros like tensor
zeros_like = torch.zeros_like(x)
zeros_like`,
        output: 
`tensor([0, 0, 0, 0, 0])`,
      }, 
      { id: 14, 
        input: 
`# Create an ones like tensor
ones_like = torch.ones_like(x)
ones_like`,
        output: 
`tensor([1, 1, 1, 1, 1])`,
      }, 
    ]
  }, 
  { id: 5, 
    name: "3_tensor_attributes",
    code: [
      { id: 15, 
        input: 
`import torch

# Create a tensor with shape 3*4
tensor_34 = torch.randn(3, 4)

# Let's check the attributes of this tensor
print(f"{tensor_34}\\n")
print(f"Shape of tensor: {tensor_34.shape}") # or tensor_34.size()
print(f"Dim Number of tensor: {tensor_34.ndim}")
print(f"Data type of tensor: {tensor_34.dtype}")
print(f"Device tensor is on: {tensor_34.device}")`,
        output: 
`tensor([[-0.9299, -1.3055,  0.6203,  1.7717],
        [-0.0485, -0.2823,  0.5894,  0.3612],
        [-0.0100,  0.9483,  0.6907,  0.4991]])

Shape of tensor: torch.Size([3, 4])
Dim Number of tensor: 2
Data type of tensor: torch.float32
Device tensor is on: cpu`,
      },
    ]
  },
  
  { id: 6, 
    name: "4.1_tensor_operations_addsub",
    code: [
      { id: 16, 
        input: 
`import torch

# Create a tensor and add a number to it
X = torch.tensor([1, 2, 3])
X + 10`,
        output: 
`tensor([11, 12, 13])`,
      },
      { id: 17, 
        input: 
`X # keep the original value unless reassign it`,
        output: 
`tensor([1, 2, 3])`,
      },
      { id: 18, 
        input: 
`# We can also use add()
# X.add(10)
torch.add(X, 10)`,
        output: 
`tensor([11, 12, 13])`,
      },
      { id: 19, 
        input: 
`# Add 2 tensors with the same shape
X + X`,
        output: 
`tensor([2, 4, 6])`,
      },
      { id: 20, 
        input: 
`torch.add(X, X)`,
        output: 
`tensor([2, 4, 6])`,
      },
      { id: 21, 
        input: 
`# Create a tensor then subtract a single number
S = torch.tensor([1, 2, 3])
S - 10`,
        output: 
`tensor([-9, -8, -7])`,
      },
      { id: 22, 
        input: 
`torch.sub(S, 10)`,
        output: 
`tensor([-9, -8, -7])`,
      },
      { id: 23, 
        input: 
`# Subtract a tensor with the same shape
S - S`,
        output: 
`tensor([0, 0, 0])`,
      },
      { id: 24, 
        input: 
`torch.sub(S, S)`,
        output: 
`tensor([0, 0, 0])`,
      },
      { id: 25, 
        input: 
`# Create a tensor then multiply it by a single number
M = torch.tensor([1, 2, 3])
M * 10`,
        output: 
`tensor([10, 20, 30])`,
      },
      { id: 26, 
        input: 
`# Can also use torch functions
torch.mul(M, 10)`,
        output: 
`tensor([10, 20, 30])`,
      },
      { id: 27, 
        input: 
`# Element-wise multiplication 
M * M`,
        output: 
`tensor([1, 4, 9])`,
      },
      { id: 28, 
        input: 
`# Can also use torch functions
torch.mul(M, M)`,
        output: 
`tensor([1, 4, 9])`,
      },
      { id: 29, 
        input: 
`# Create a tensor then divide it by a single number
D = torch.tensor([1, 2, 3])
D / 10`,
        output: 
`tensor([1.1000, 1.2000, 1.3000])`,
      },
      { id: 30, 
        input: 
`# Can also use torch function
torch.div(D, 10) # build-in function`,
        output: 
`tensor([1.1000, 1.2000, 1.3000])`,
      },
      { id: 31, 
        input: 
`# Divide it by a tensor (same shape)
D / D`,
        output: 
`tensor([1., 1., 1.])`,
      },
      { id: 32, 
        input: 
`# Divide with div()
torch.div(D, D)`,
        output: 
`tensor([1., 1., 1.])`,
      },
    ]
  },
  { id: 7, 
    name: "4.2_tensor_operations_matmul",
    code: [
      { id: 33, 
        input: 
`import torch

X = torch.tensor([1, 2, 3]) # vector`,
        output: 
`tensor([1, 2, 3])`,
      },
      { id: 34, 
        input: 
`# Matrix Multiplication
X.matmul(X) # dot product`,
        output: 
`tensor(14)`,
      },
      { id: 35, 
        input: 
`# Prefer '@', clear and concise
X @ X`,
        output: 
`tensor(14)`,
      },
      { id: 36, 
        input: 
`%%time
# Matrix multiplication by hand
# (avoid doing operations with for loops at all cost, they are computationally expensive)
value = 0
for i in range(len(X)):
  value += X[i] * X[i]
value`,
        output: 
`CPU times: user 995 µs, sys: 0 ns, total: 995 µs
Wall time: 2.22 ms
tensor(14)`,
      },
      { id: 37, 
        input: 
`%%time
torch.matmul(X, X)`,
        output: 
`CPU times: user 138 µs, sys: 0 ns, total: 138 µs
Wall time: 142 µs
tensor(14)`,
      },
      { id: 38, 
        input: 
`# Create 2 matrices, and matrix multiplication
A = torch.tensor([[1, 2, 0],
                  [0, 1, 0]])

B = torch.tensor([[2, 1],
                  [0, 1],
                  [6, 8]])

C = A @ B
C`,
        output: 
`tensor([[2, 3],
        [0, 1]])`,
      },
      { id: 39, 
        input: 
`# Check the tensors' shape
print(f"Shape of A: {A.shape}")
print(f"Shape of B: {B.shape}\\n")
print(f"Shape of C: {C.shape}")`,
        output: 
`Shape of A: torch.Size([2, 3])
Shape of B: torch.Size([3, 2])

Shape of C: torch.Size([2, 2])`,
      },
    ]
  },
  { id: 8, 
    name: "4.3_tensor_operations_aggregate",
    code: [
      { id: 40, 
        input: 
`# Create a tensor
x = torch.arange(0, 50, 10)
x, x.dtype`,
        output: 
`(tensor([ 0, 10, 20, 30, 40]), torch.int64)`,
      },
      { id: 41, 
        input: 
`print(f"Minimum: {x.min()}")
print(f"Maximum: {x.max()}")
print(f"Sum: {x.sum()}")`,
        output: 
`Minimum: 0
Maximum: 40
Sum: 100`,
      },
      { id: 42, 
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
      { id: 43, 
        input: 
`# mean(): won't work without float datatype
# Let's specify to 'float32', we'll cover type() next section
print(f"Mean: {x.type(torch.float32).mean()}")`,
        output: 
`Mean: 20.0`,
      },
      { id: 44, 
        input: 
`print(f"Minimum: {torch.min(x)}")
print(f"Maximum: {torch.max(x)}")
print(f"Mean: {torch.mean(x.type(torch.float32))}")
print(f"Sum: {torch.sum(x)}")`,
        output: 
`Minimum: 0
Maximum: 40
Mean: 20.0
Sum: 100`,
      },
      { id: 45, 
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
  { id: 9, 
    name: "5.1_tensor_manipulate_shape",
    code: [
      { id: 46, 
        input: 
`import torch

# Create a tensor and check its data type
x = torch.tensor([0., 10., 20., 30., 40.])

# Check the tensor shape
x.shape`,
        output: 
`torch.Size([5])`,
      },
      { id: 47, 
        input: 
`# unsqueeze(): adds a single dimension to a target tensor at a specific dim
print(f"Previous tensor:\\n {x, x.shape}\\n")

# Add an extra dimension with unsqueeze()
x_unsqueezed = x.unsqueeze(dim=0)

print(f"Squeezed tensor:\\n {x_unsqueezed, x_unsqueezed.shape}")`,
        output: 
`Previous tensor:
 (tensor([ 0., 10., 20., 30., 40.]), torch.Size([5]))

Squeezed tensor:
 (tensor([[ 0., 10., 20., 30., 40.]]), torch.Size([1, 5]))`,
      },
      { id: 48, 
        input: 
`# squeeze(): remove all single dimensions from a target tensor
print(f"Previous tensor:\\n {x_unsqueezed, x_unsqueezed.shape}]\\n")

# Remove extra dimensions from x_reshaped
x_squeezed = x_unsqueezed.squeeze()

print(f"Squeezed tensor:\\n {x_squeezed, x_squeezed.shape}")`,
        output: 
`Previous tensor:
 (tensor([[ 0., 10., 20., 30., 40.]]), torch.Size([1, 5]))

Squeezed tensor:
 (tensor([ 0., 10., 20., 30., 40.]), torch.Size([5]))`,
      },
    ]
  },
  { id: 10, 
    name: "tensor_manipulate_transpose",
    code: [
      { id: 49, 
        input: 
`import torch

# Create 2 tensors and matrix Multiplication
A = torch.tensor([[1., 2., 1.], 
                  [0., 1., 0.]])
B = torch.tensor([[1., 2., 3.],
                  [4., 5., 6.]])

torch.matmul(A, B) # error's here`,
        output: 
`---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
<ipython-input-15-2e352ffdc080> in <cell line: 9>()
      7                   [4., 5., 6.]])
      8 
----> 9 torch.matmul(A, B) # error's here

RuntimeError: mat1 and mat2 shapes cannot be multiplied (2x3 and 2x3)`,
      },
      { id: 50, 
        input: 
`# Their inner shapes are not matched
print(f"A Shape: {A.shape}")
print(f"B Shape: {B.shape}\\n")

# Transpose B to match the inner dim
print(f"Original B shape: {B.shape}")
print(f"Transposed B shape: {B.T.shape}\\n")

# A @ B.T
C = torch.matmul(A, B.T)

print(f"Matrix Multiplication :\\n {A.shape} * {B.T.shape} <- inner dim match")
print(f"Output:")
print(f"C shape: {C.shape}")`,
        output: 
`Shape of A: torch.Size([2, 3])
Shape of B: torch.Size([2, 3])

Original B shape: torch.Size([2, 3])
Transposed B shape: torch.Size([3, 2])

Matrix Multiplication :
torch.Size([2, 3]) * torch.Size([3, 2]) <- inner dim match
Output:
C shape: torch.Size([2, 2])`,
      },
      { id: 51, 
        input: 
`# transpose(input, dim0, dim1)
B_transposed = torch.transpose(B, 0, 1)

print(f"Original B shape: {B.shape}")
print(f"Transposed B shape: {B_transposed.shape}\\n")

result = A @ B_transposed
print(f"Matrix Multiplication :\\n {A.shape} * {B_transposed.shape} <- inner dim match")
print(f"Output:")
print(f"result shape: {result.shape}")`,
        output: 
`Original B shape: torch.Size([2, 3])
Transposed B shape: torch.Size([3, 2])

Matrix Multiplication :
torch.Size([2, 3]) * torch.Size([3, 2]) <- inner dim match
Output:
result shape: torch.Size([2, 2])`,
      },
    ]
  },
  { id: 11, 
    name: "tensor_manipulate_permuate",
    code: [
      { id: 52, 
        input: 
`# Create a tensor with a specific shape
p = torch.rand(size=[128, 128, 3])

print(f"Original tensor shape: {p.shape}")

# permute(): rearrange the axis order
p_permuted = p.permute(2, 0, 1)
print(f"Permuted tensor shape: {p_permuted.shape}")`,
        output: 
`Original tensor shape: torch.Size([128, 128, 3])
Permuted tensor shape: torch.Size([3, 128, 128])`,
      },
      { id: 53,
        input:
`# Create a 3 dimensional tensor with shape [3, 4, 5]
P = torch.randint(low=0, high=10, size=[3, 4, 5], dtype=torch.int8)
P, P.shape`,
        output:
`(tensor([[[3, 3, 2, 7, 9],
          [8, 1, 0, 3, 8],
          [3, 0, 2, 5, 5],
          [4, 5, 3, 8, 5]],
 
         [[9, 3, 3, 8, 8],
          [0, 3, 3, 2, 1],
          [2, 4, 6, 0, 5],
          [5, 6, 6, 0, 5]],
 
         [[7, 8, 4, 3, 7],
          [6, 1, 5, 1, 9],
          [7, 2, 8, 3, 5],
          [1, 0, 9, 2, 2]]], dtype=torch.int8),
 torch.Size([3, 4, 5]))`,
      },
      { id: 54,
        input:
`# Let's permute this tensor
P_permuted_1 = P.permute(0, 2, 1) # rearrange dim order with index
P_permuted_1, P_permuted_2.shape`,
        output:
`(tensor([[[3, 8, 3, 4],
          [3, 1, 0, 5],
          [2, 0, 2, 3],
          [7, 3, 5, 8],
          [9, 8, 5, 5]],
 
         [[9, 0, 2, 5],
          [3, 3, 4, 6],
          [3, 3, 6, 6],
          [8, 2, 0, 0],
          [8, 1, 5, 5]],
 
         [[7, 6, 7, 1],
          [8, 1, 2, 0],
          [4, 5, 8, 9],
          [3, 1, 3, 2],
          [7, 9, 5, 2]]], dtype=torch.int8),
 torch.Size([5, 3, 4]))`,
      },
      { id: 55,
        input:
`P_permuted_2 = P.permute(2, 0, 1)
P_permuted_2, P_permuted_2.shape`,
        output:
`(tensor([[[3, 8, 3, 4],
          [9, 0, 2, 5],
          [7, 6, 7, 1]],
 
         [[3, 1, 0, 5],
          [3, 3, 4, 6],
          [8, 1, 2, 0]],
 
         [[2, 0, 2, 3],
          [3, 3, 6, 6],
          [4, 5, 8, 9]],
 
         [[7, 3, 5, 8],
          [8, 2, 0, 0],
          [3, 1, 3, 2]],
 
         [[9, 8, 5, 5],
          [8, 1, 5, 5],
          [7, 9, 5, 2]]], dtype=torch.int8),
 torch.Size([5, 3, 4]))`,
      },
      { id: 56,
        input:
`P_permuted_3 = P.permute(2, 1, 0)
P_permuted_3, P_permuted_3.shape`,
        output:
`(tensor([[[3, 9, 7],
          [8, 0, 6],
          [3, 2, 7],
          [4, 5, 1]],
 
         [[3, 3, 8],
          [1, 3, 1],
          [0, 4, 2],
          [5, 6, 0]],
 
         [[2, 3, 4],
          [0, 3, 5],
          [2, 6, 8],
          [3, 6, 9]],
 
         [[7, 8, 3],
          [3, 2, 1],
          [5, 0, 3],
          [8, 0, 2]],
 
         [[9, 8, 7],
          [8, 1, 9],
          [5, 5, 5],
          [5, 5, 2]]], dtype=torch.int8),
 torch.Size([5, 4, 3]))`,
      }
    ]
  },
  { id: 12, 
    name: "tensor_manipulate_reshape",
    code: [
      { id: 57, 
        input: 
`import torch

x = torch.tensor([0, 10, 20, 30, 40])

# add an extra dim with reshape()
x_1 = x.reshape(5, 1) # add an inner dim
x_2 = x.reshape(1, 5) # add an outer dim

print(x_1)
print(x_2)`,
        output: 
`tensor([[ 0],
        [10],
        [20],
        [30],
        [40]])
tensor([[ 0, 10, 20, 30, 40]])`,
      },
      { id: 58, 
        input: 
`print(f"Original tensor:\\n {x}\\n")

x_2[:, 0] = 99

# change x_2, x changed, cus sharing the same data..
# if necessary it might create a new copy of the data
print(f"Change the reshaped tensor:\\n {x_2}\\n")
print(f"Original tensor changed:\\n {x}")`,
        output: 
`Original tensor:
 tensor([99, 10, 20, 30, 40])

Change the reshaped tensor:
 tensor([[99, 10, 20, 30, 40]])

Original tensor changed:
 tensor([99, 10, 20, 30, 40])`,
      },
      { id: 59, 
        input: 
`# change tensor shape with view()
# view(): a different way of looking at the same data
x_3 = x.view(5, 1)
x_4 = x.view(1, 5)

x_4[:, 0] = 88

print(f"Change the view (inner dim):\\n {x_3}")
print(f"Change the view (outer dim):\\n {x_4}\\n")
print(f"Original tensor:\\n {x}")`,
        output: 
`Change the view (inner dim):
 tensor([[88],
        [10],
        [20],
        [30],
        [40]])
Change the view (outer dim):
 tensor([[88, 10, 20, 30, 40]])

Original tensor:
 tensor([88, 10, 20, 30, 40])`,
      },
      { id: 60, 
        input: 
`# Create a 3d tensor
P = torch.randint(low=0, high=10, size=[3, 4, 5], dtype=torch.int8)
P, P.shape`,
        output:
`(tensor([[[6, 3, 0, 3, 6],
          [0, 3, 7, 3, 8],
          [9, 8, 2, 5, 5],
          [4, 8, 1, 7, 9]],
 
         [[2, 6, 8, 2, 8],
          [2, 1, 0, 0, 6],
          [5, 0, 6, 9, 2],
          [4, 2, 0, 2, 3]],
 
         [[3, 1, 6, 4, 6],
          [9, 4, 8, 9, 2],
          [5, 9, 9, 6, 4],
          [0, 8, 2, 4, 0]]], dtype=torch.int8),
 torch.Size([3, 4, 5]))`,
      },
      { id: 61, 
        input: 
`# with reshape(), we can rearrange this tensor
P.reshape(5, 3, 4)
# P.view(5, 3, 4) # as reshape()`,
        output:
`tensor([[[6, 3, 0, 3],
         [6, 0, 3, 7],
         [3, 8, 9, 8]],

        [[2, 5, 5, 4],
         [8, 1, 7, 9],
         [2, 6, 8, 2]],

        [[8, 2, 1, 0],
         [0, 6, 5, 0],
         [6, 9, 2, 4]],

        [[2, 0, 2, 3],
         [3, 1, 6, 4],
         [6, 9, 4, 8]],

        [[9, 2, 5, 9],
         [9, 6, 4, 0],
         [8, 2, 4, 0]]], dtype=torch.int8)`,
      },
      { id: 62, 
        input: 
`# 3d -> 2d, 
# in deep learning, we often use to flatten a tensor
P.reshape(5, 12)
# P.view(5, 12)`,
        output:
`tensor([[6, 3, 0, 3, 6, 0, 3, 7, 3, 8, 9, 8],
        [2, 5, 5, 4, 8, 1, 7, 9, 2, 6, 8, 2],
        [8, 2, 1, 0, 0, 6, 5, 0, 6, 9, 2, 4],
        [2, 0, 2, 3, 3, 1, 6, 4, 6, 9, 4, 8],
        [9, 2, 5, 9, 9, 6, 4, 0, 8, 2, 4, 0]], dtype=torch.int8)`,
      },
      { id: 63, 
        input: 
`# 3d -> 4d, also often used in deep learning model
P.reshape(2, 2, 3, 5)
# P.view(2, 2, 3, 5)`,
        output:
`tensor([[[[6, 3, 0, 3, 6],
          [0, 3, 7, 3, 8],
          [9, 8, 2, 5, 5]],

         [[4, 8, 1, 7, 9],
          [2, 6, 8, 2, 8],
          [2, 1, 0, 0, 6]]],


        [[[5, 0, 6, 9, 2],
          [4, 2, 0, 2, 3],
          [3, 1, 6, 4, 6]],

         [[9, 4, 8, 9, 2],
          [5, 9, 9, 6, 4],
          [0, 8, 2, 4, 0]]]], dtype=torch.int8)`,
      },
    ]
  },
  { id: 13, 
    name: "5.2_tensor_manipulate_dtype",
    code: [
      { id: 64, 
        input: 
`import torch

# Create a tensor and check its data type
A = torch.arange(1.0, 8.0)
B = torch.ones(5)
print(f"{A,A.dtype}")
print(f"{B,B.dtype}")`,
        output: 
`(tensor([1., 2., 3., 4., 5., 6., 7.]), torch.float32)
(tensor([1., 1., 1., 1., 1.]), torch.float32)`,
      },
      { id: 65, 
        input: 
`# Change data type of B
C =B.type(torch.float16)

print(B, B.dtype)
print(C)`,
        output: 
`tensor([1., 1., 1., 1., 1.]) torch.float32
tensor([1., 1., 1., 1., 1.], dtype=torch.float16)`,
      },
      { id: 66, 
        input: 
`import torch
import numpy as np

# Numpy Array -> Tensor
array = np.arange(1.0, 8.0)
tensor = torch.from_numpy(array)

print(array, array.dtype) # float64
print(tensor) # tensor: float32 -> float64`,
        output: 
`[1. 2. 3. 4. 5. 6. 7.] float64
tensor([1., 2., 3., 4., 5., 6., 7.], dtype=torch.float64)`,
      },
      { id: 67, 
        input: 
`# Change the array, keep the tensor
array = array + 1

print(array, array.dtype)
print(tensor, tensor.dtype)`, 
        output: 
`[ 7.  8.  9. 10. 11. 12. 13.] float64
tensor([1., 1., 1., 1., 1., 1., 1.]) torch.float32`,
      },
      { id: 68, 
        input: 
`# Tensor -> NumPy array
tensor = torch.ones(7)
numpy_tensor = tensor.numpy()

print(tensor, tensor.dtype) # float32
print(numpy_tensor, numpy_tensor.dtype) # float64 -> float32`, 
        output: 
`tensor([1., 1., 1., 1., 1., 1., 1.]) torch.float32
[1. 1. 1. 1. 1. 1. 1.] float32`,
      },
      { id: 69, 
        input: 
`# Change the tensor, what happens to numpy_tensor?
tensor = tensor + 1
tensor, numpy_tensor

print(tensor, tensor.dtype)
print(numpy_tensor, numpy_tensor.dtype)`, 
        output: 
`tensor([2., 2., 2., 2., 2., 2., 2.]) torch.float32
[1. 1. 1. 1. 1. 1. 1.] float32`,
      },
    ]
  }, 
  { id: 14, 
    name: "5.3_tensor_manipulate_catstack",
    code: [
      { id: 70, 
        input: 
`import torch

# Create 2 tensors
A = torch.tensor([1., 2., 3.])
B = torch.tensor([2., 3., 4.])

print(f"{A, A.shape}")
print(f"{B, B.shape}\\n")`,
        output: 
`(tensor([1., 2., 3.]), torch.Size([3]))
(tensor([2., 3., 4.]), torch.Size([3]))`,
      },
      { id: 71, 
        input: 
`# Concatenate A and B on the 1st dim
A_cat_B_1 = torch.cat([A, B], dim=0)
print(f"Concatenated tensor:\\n {A_cat_B_1, A_cat_B_1.shape}\\n")`,
        output: 
`Concatenated tensor:
 (tensor([1., 2., 3., 2., 3., 4.]), torch.Size([6]))`,
      },
      { id: 72, 
        input: 
`# Stack A and B on the 1st dim
A_stack_B_1 = torch.stack([A, B], dim=0)
print(f"Stacked tensor:\\n {A_stack_B_1, A_stack_B_1.shape}\\n")

# Stack A and B on the 2nd dim
A_stack_B_2 = torch.stack([A, B], dim=1)
print(f"Stacked tensor:\\n {A_stack_B_2, A_stack_B_2.shape}")`,
        output: 
`Stacked tensor:
 (tensor([[1., 2., 3.],
        [2., 3., 4.]]), torch.Size([2, 3]))

Stacked tensor:
 (tensor([[1., 2.],
        [2., 3.],
        [3., 4.]]), torch.Size([3, 2]))`,
      },
    ]
  },
  { id: 15,
    name: ":: Stack tensors",
    code: [
      { id: 73,
        input: 
`import torch

# Create two 2d tensors
A = torch.randint(low=0, high=10, size=[4, 5], dtype=torch.int8)
B = torch.randint(low=0, high=10, size=[4, 5], dtype=torch.int8)

print(f"{A, A.shape}\\n")
print(B, B.shape)`,
        output:
`(tensor([[7, 6, 3, 5, 2],
        [4, 6, 6, 2, 9],
        [1, 0, 8, 9, 6],
        [0, 6, 6, 3, 7]], dtype=torch.int8), torch.Size([4, 5]))

tensor([[5, 1, 5, 6, 1],
        [9, 9, 7, 2, 3],
        [3, 7, 8, 3, 0],
        [7, 8, 3, 3, 9]], dtype=torch.int8) torch.Size([4, 5])`,
      },
      { id: 74,
        input: 
`# stack 2 tensors on the 1st dim (depth dim added)
C = torch.stack((A, B), dim=0)
C, C.shape`,
        output:
`(tensor([[[7, 6, 3, 5, 2],
          [4, 6, 6, 2, 9],
          [1, 0, 8, 9, 6],
          [0, 6, 6, 3, 7]],
 
         [[5, 1, 5, 6, 1],
          [9, 9, 7, 2, 3],
          [3, 7, 8, 3, 0],
          [7, 8, 3, 3, 9]]], dtype=torch.int8),
 torch.Size([2, 4, 5]))`,
      },
    ]
  },
  { id: 16,
    name: ":: Stack tensors",
    code: [
      { id: 75,
        input: 
`# Concatenate A and B o the 1st dim (ndim no changing)
C = torch.cat((A, B), dim=0)
C, C.shape`,
        output:
`(tensor([[7, 6, 3, 5, 2],
         [4, 6, 6, 2, 9],
         [1, 0, 8, 9, 6],
         [0, 6, 6, 3, 7],
         [5, 1, 5, 6, 1],
         [9, 9, 7, 2, 3],
         [3, 7, 8, 3, 0],
         [7, 8, 3, 3, 9]], dtype=torch.int8),
 torch.Size([8, 5]))`,
      },
      { id: 76,
        input: 
`# with vstack(), we can get the same result
D = torch.vstack((A, B))
D, D.shape`,
        output:
`(tensor([[7, 6, 3, 5, 2],
         [4, 6, 6, 2, 9],
         [1, 0, 8, 9, 6],
         [0, 6, 6, 3, 7],
         [5, 1, 5, 6, 1],
         [9, 9, 7, 2, 3],
         [3, 7, 8, 3, 0],
         [7, 8, 3, 3, 9]], dtype=torch.int8),
 torch.Size([8, 5]))`,
      },
    ]
  },
  { id: 17,
    name: ":: Stack tensors",
    code: [
      { id: 77,
        input: 
`# Concatenate A and B o the 2nd dim (ndim no changing)
C = torch.cat((A, B), dim=1)
C, C.shape`,
        output:
`(tensor([[7, 6, 3, 5, 2, 5, 1, 5, 6, 1],
         [4, 6, 6, 2, 9, 9, 9, 7, 2, 3],
         [1, 0, 8, 9, 6, 3, 7, 8, 3, 0],
         [0, 6, 6, 3, 7, 7, 8, 3, 3, 9]], dtype=torch.int8),
 torch.Size([4, 10]))`,
      },
      { id: 78,
        input: 
`# with stack(), we can get the same result
D = torch.hstack((A, B))
D, D.shape`,
        output:
`(tensor([[7, 6, 3, 5, 2, 5, 1, 5, 6, 1],
         [4, 6, 6, 2, 9, 9, 9, 7, 2, 3],
         [1, 0, 8, 9, 6, 3, 7, 8, 3, 0],
         [0, 6, 6, 3, 7, 7, 8, 3, 3, 9]], dtype=torch.int8),
 torch.Size([4, 10]))`,
      },
    ]
  },
  { id: 18, 
    name: "6.1_tensor_index_basicIndexing",
    code: [
      {
        id: 79, 
        input: 
`import torch

# Create a tensor
A = torch.arange(0, 9).reshape(1, 3, 3)
A, A.shape`,
        output: 
`(tensor([[[0, 1, 2],
          [3, 4, 5],
          [6, 7, 8]]]),
 torch.Size([1, 3, 3]))`,
      },
      {
        id: 80, 
        input: 
`# Let's index this tensor by bracket
print(f"The outer square bracket: \\n{A[0]}")
print(f"The middle squre bracket: {A[0][0]}")
print(f"The inner square bracket: {A[0][0][0]}")`,
        output: 
`The outer square bracket: 
tensor([[0, 1, 2],
        [3, 4, 5],
        [6, 7, 8]])
The middle squre bracket: tensor([0, 1, 2])
The inner square bracket: 0`,
      },
      {
        id: 81, 
        input: 
`# Another way to index tensor with ','
print(f"The outer square bracket: \\n{A[0]}")
print(f"The middle squre bracket: {A[0, 0]}")
print(f"The inner square bracket: {A[0, 0, 0]}")`,
        output: 
`The outer square bracket: 
tensor([[0, 1, 2],
        [3, 4, 5],
        [6, 7, 8]])
The middle squre bracket: tensor([0, 1, 2])
The inner square bracket: 0`,
      },
      {
        id: 82, 
        input: 
`# Index a particular number('8')
A[0][2][2]`,
        output: 
`tensor(8)`,
      },
      {
        id: 83, 
        input: 
`A[0, 2, 2]`,
        output: 
`tensor(8)`,
      },
      {
        id: 84, 
        input: 
`A[:, 2, 2] # ':' - means all values in current dim`,
        output: 
`tensor([8])`,
      },
      {
        id: 85, 
        input: 
`# Row/Column Indexing
A[:, 0]`,
        output: 
`tensor([[0, 1, 2]])`,
      },
      {
        id: 86, 
        input: 
`A[:, :, 1]`,
        output: 
`tensor([[1, 4, 7]])`,
      },
    ]
  },
  { id: 16, 
    name: "6.2_tensor_index_sliceboolean",
    code: [
      {
        id: 87,
        input: 
`# Slicing Indexing
# (extract a range of elements form a tensor)
A[:, :2, :2] # ':2" - means get the 1st 2 elements in current dim`,
        output: 
`tensor([[[0, 1],
         [3, 4]]])`,
      },
      {
        id: 88,
        input: 
`# (specific a step value in a tensor)
A[:, :2, ::2] # '::2' - means take every 2nd element`,
        output: 
`tensor([[[0, 2],
         [3, 5]]])`,
      },
      {
        id: 89,
        input: 
`# Boolean Indexing
# (with a condition to index tensor)
A[A > 3]`,
        output: 
`tensor([4, 5, 6, 7, 8])`,
      },
    ]
  },
  { id: 17, 
    name: "7.1_tensor_reproducibility_randomness",
    code: [
      { id: 90,
        input: 
`import torch

# Create 2 random tensors
A = torch.rand(3, 2)
B = torch.rand(3, 2)

print(f"Tensor A: \\n{A}")
print(f"Tensor B: \\n{B}\\n")
print(f"Does Tensor A equal Tensor B?")
A == B`,
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
tensor([[Falso, Falso],
        [Falso, Falso],
        [Falso, Falso]])`,
       },
    ]
  },
  { id: 18,
    name: "7.3_tensor_randomseed",
    code: [
      { id: 91, 
        input:
`# Let's make some random but reproducible tensors
import torch

# Set random seed fro reproducibility
RANDOM_SEED = 69 # you could choose any number here

torch.manual_seed(RANDOM_SEED)
A = torch.rand(3, 4)

torch.manual_seed(RANDOM_SEED)
B = torch.rand(3, 4)

print(f"Tensor A: \\n{random_tensor_A}\\n")
print(f"Tensor B: \\n{B}\\n")
print(f"Does Tensor A equal Tensor B?")
random_tensor_A == B`,
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
  { id: 19,
    name: "tensor_gpus",
    code: [
      { id: 92,
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
       { id: 93,
        input: 
`# check if GPU is available
import torch
torch.cuda.is_available()`,
        output:
`TRUE`,
       },
       { id: 94,
        input: 
`# Set Device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
device`,
        output:
`'cuda'`,
       },
       { id: 95,
        input: 
`# Count number of devices
torch.cuda.device_count()`,
        output:
`1`,
       },
       { id: 96,
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
       { id: 97,
        input: 
`# If tensor is on GPU, we can't transform it to NumPy
# cus NumPy array don't support GPU
tensor_on_gpu.numpy()`,
        output:
`-------------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-64-53175578f49e> in <cell line: 2>()
      1 # If tensor is on GPU, can't transform it to NumPy (error)
----> 2 tensor_on_gpu.numpy()

TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.`,
       },
       { id: 98,
        input: 
`# Set the tensor back to cpu, then copy it to NumPy
tensor_on_cpu = tensor_on_gpu.cpu().numpy()
tensor_on_cpu`,
        output:
`array([1, 2, 3])`,
       },
       { id: 99,
        input: 
`# the original tensor is still on GPU
tensor_on_gpu`,
        output:
`tensor([1, 2, 3], device='cuda:0')`,
       },
    ]
  }
]