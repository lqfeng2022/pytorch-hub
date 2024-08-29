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
  }
]