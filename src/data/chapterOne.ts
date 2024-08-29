export default [
  { id: 1, 
    name: "TENSOR",
    description: "",
    sections: [
      { id: 1,
        name: "TENSOR", 
        value: "TENSOR is a multi-dimensional array with a single data type.",
        image: "",
        content: [
          { id: 1, 
            title: "",
            value: "A tensor is a mathematical object that generalizes the concepts of scalars, vectors, and matrices to higher dimensions."
          },
          { id: 2, 
            title: "",
            value: "Tensors are a fundamental tool in various fields, including physics, engineering, and machine learning, cus they can represent complex, multi-dimensional data in a structured way."
          },
        ]
      },
      { id: 2,
        name: "scalar, vector, MATRIX, TENSOR", 
        value: "",
        image: "src/assets/chapter_one/tensors.jpeg",
        content: [
          { id: 1, 
            title: "scalar",
            value: "Scalar is a single number (0-dimensional tensor), we represent it as a point. As a common practice, we use lowercase for scalars."
          },
          { id: 2, 
            title: "vector",
            value: "Vector is a list of numbers (1-dimensional tensor), we represent it as a straight line. As a common practice, we use lowercase for vectors and bold it."
          },
          { id: 3, 
            title: "MATRIX",
            value: "Matrix is a grid of numbers (2-dimensional tensor), we represent it as a square or rectangle. In our example, there are 3 rows and each row own 3 column of number. As a common practice, we use uppercase for matrix."
          },
          { id: 4, 
            title: "TENSOR",
            value: "Tensor can extend to three or more dimensions. For example, a 3D tensor might represent a block of data, like a stack of matrices, and a 4D tensor could represent data that varies over time, like a video. As a common practice, we use uppercase for Tensor and bold it."
          },
        ]
      },
    ]
  },
  { id: 2, 
    name: "Create TENSORs",
    description: "",
    sections: [
      { id: 1,
        name: "2.1 Random Tensor", 
        value: "",
        image: "src/assets/chapter_one/randomtensor.jpeg",
        content: [
          { id: 1, 
            title: "What's Random Tensor?",
            value: "A random tensor is simply a tensor filled with random numbers. These numbers are typically drawn from a certain probability distribution, like a Normal Distribution — where numbers are more likely to be close to the mean, or a Uniform Distribution — where all numbers in a range have an equal chance of appearing."
          },
          { id: 2, 
            title: "Why Random Tensors?",
            value: "Random tensor are often used in machine learning and deep learning for various reasons, like initializing Weights. When you're setting up a neural network, the weights of the network's layers need to start with some initial values. These initial weights are usually set to small random numbers. Starting with random values helps the learning process begin in a way that prevents symmetry, allowing the network to learn effectively.."
          },
          { id: 3, 
            title: "How Create Random Tensors?",
            value: "In PyTorch, you can create random tensors using functions like torch.rand() for uniform distributions, torch.randn() for normal distributions, and others, depending on your needs."
          },
        ]
      },
      { id: 2,
        name: "2.2 Zeros/Ones Tensors", 
        value: "",
        image: "src/assets/chapter_one/zerosOnes.jpeg",
        content: [
          { id: 1, 
            title: "What's Zeros/Ones Tensors?",
            value: "A zeros/ones tensor is a tensor where every element is initialized to zeros/ones. It's like an array filled with zeros/ones but can be multi-dimensional and more complex."
          },
          { id: 2, 
            title: "Why Zeros/Ones Tensors?",
            value: "In neural networks, bias terms are often initialized to zero, cus this doesn't introduce any asymmetry or preference during training. And sometimes you might want to initialize certain parameters or tensors to one instead of zero, especially if you're dealing with multiplicative operations. There are many other reasons, we'll cover it when we use it in a particular scenario."
          },
          { id: 3, 
            title: "How Create Zeros/Ones Tensors?",
            value: "You can create zeros/ones tensor in PyTorch using torch.zeros()/torch.ones(). For example, torch.zeros(3, 3)/torch.ones(3, 3) would create a 3x3 matrix full of zeros/ones."
          },
        ]
      },
      { id: 3,
        name: "2.3 A Range Tensor", 
        value: "",
        image: "src/assets/chapter_one/range.jpeg",
        content: [
          { id: 1, 
            title: "What's Range Tensor?",
            value: "A range tensor is a tensor that contains a sequence of numbers in a specified range, with a specified step size. It's similar to Python's built-in range() function, but instead of generating a list, it generates a tensor."
          },
          { id: 2, 
            title: "Why use Range of Tensor?",
            value: "Range tensors are often used for creating index values when slicing or performing operations on specific ranges within other tensors.."
          },
          { id: 3, 
            title: "How Create a Range of Tensor?",
            value: "In PyTorch, you can create a range tensor using functions like torch.arange(). And the start means the starting value of the sequence (inclusive), end means the ending value of the sequence (exclusive), and step means the difference between each consecutive value."
          },
        ]
      },
      { id: 4,
        name: "2.4 A Tensor Like", 
        value: "",
        image: "src/assets/chapter_one/likes.jpeg",
        content: [
          { id: 1, 
            title: "What's Tensor Like?",
            value: "When we talk about “tensor like” in PyTorch, we're usually referring to functions like torch.zeros_like() and torch.ones_like(). These functions are used to create new tensors that have the same shape and device as a given tensor but are filled with zeros or ones."
          },
          { id: 2, 
            title: "How Create a Tensor Like?",
            value: "In PyTorch we use torch.zeros_like() and torch.ones_like() to create Tensor like."
          },
          { id: 3, 
            title: "Why use Tensor Like?",
            value: "Sometimes, in deep learning model, we need to create a new tensor with the same shape and device as the input tensor, and Tensor Like is an efficient way"
          },
        ]
      },
    ]
  },
  { id: 3, 
    name: "TENSOR Attributes",
    description: "",
    sections: [
      { id: 1,
        name: "3. TENSOR Attributes", 
        value: "In PyTorch, tensors have several important attributes that provide information about their structure, data type, and where they are stored (CPU or GPU). The main attributes you'll often work with are shape, ndim, dtype, and device. Here's a brief overview of each:",
        image: "src/assets/chapter_one/attributes.jpeg",
        content: [
          { id: 1, 
            title: "Shape",
            value: "The shape attribute tells you the size of each dimension of the tensor. It's a tuple of integers where each integer represents the size of the tensor along that dimension. Knowing the shape of a tensor is crucial because it defines the structure of the data you're working with. It tells you how many elements are in each dimension."
          },
          { id: 2, 
            title: "ndim",
            value: "The ndim attribute (also known as the number of dimensions or rank) gives you the number of dimensions (axes) of the tensor. Understanding the number of dimensions is essential for operations like reshaping, broadcasting, and when you're visualizing the structure of data."
          },
          { id: 3, 
            title: "dtype",
            value: "The dtype attribute indicates the data type of the elements stored in the tensor, like torch.float32, torch.int64, etc. The data type determines the precision and range of the numbers in the tensor. Choosing the right data type can affect both performance and memory usage."
          },
          { id: 4, 
            title: "device",
            value: "The device attribute tells you where the tensor's data is stored: either on the CPU or a GPU. Tensors on a GPU will have their device set to something like cuda:0. Knowing where a tensor is stored is crucial when you're working with GPUs for accelerated computation. You need to ensure that all tensors involved in operations are on the same device to avoid errors."
          },
        ]
      },
      { id: 2,
        name: "3.1 SHAPE", 
        value: "Working with the shape attribute in PyTorch can sometimes be tricky, especially when you're manipulating tensors or performing operations that involve broadcasting, reshaping, or matrix multiplication. Let's dive a bit deeper into what shape is and how to avoid common mistakes.",
        image: "src/assets/chapter_one/shape.jpeg",
        content: [
          { id: 1, 
            title: "Definition",
            value: "The shape of a tensor is a tuple that describes the dimensions of the tensor. Each element in the tuple represents the size of the tensor along that particular dimension."
          },
          { id: 2, 
            title: "Example",
            value: "A tensor with shape (3, 4) has 3 rows and 4 columns. If you think of it like a matrix, it's a 2D tensor with 3 rows and 4 columns."
          },
        ]
      },
    ]
  },
  { id: 4, 
    name: "TENSOR Operations",
    description: "",
    sections: [
      { id: 1,
        name: "4. TENSOR Operations", 
        value: "",
        image: "src/assets/chapter_one/operations.jpeg",
        content: [
          { id: 1, 
            title: "",
            value: "Tensor operations are mathematical operations that can be performed on tensors, similar to how we perform operations on scalars, vectors, and matrices. These operations are essential in machine learning, where tensors are used to represent complex, multi-dimensional data."
          },
        ]
      },
      { id: 2,
        name: "4.1 Add/Subtract Operations", 
        value: "",
        image: "src/assets/chapter_one/addSub.jpeg",
        content: [
          { id: 1, 
            title: "",
            value: "You can add or subtract a number or two tensors of the same shape by performing the operation element by element."
          },
        ]
      },
      { id: 3,
        name: "4.2 Multiplication/Division Operations", 
        value: "",
        image: "src/assets/chapter_one/mulDiv.jpeg",
        content: [
          { id: 1, 
            title: "",
            value: "Similar to addition and subtraction, you can multiply or divide a number or tensors element by element."
          },
        ]
      },
      { id: 4,
        name: "4.3 Matrix Multiplication", 
        value: "Matrix multiplication is a fundamental operation in linear algebra where two matrices are multiplied to produce a third matrix. This operation is widely used in various fields, including computer science, engineering, physics, and machine learning.",
        image: "src/assets/chapter_one/matmul.jpeg",
        content: [
          { id: 1, 
            title: "",
            value: "Matrix multiplication, or dot product, which is the sum of the products of corresponding elements from the row and column. And the number of columns in the 1st matrix must be equal to the number of rows in the 2nd matrix."
          },
        ]
      },
      { id: 5,
        name: "4.4 Dot Product", 
        value: "",
        image: "src/assets/chapter_one/dot.jpeg",
        content: [
          { id: 1, 
            title: "",
            value: "The dot product is a fundamental operation in vector algebra, commonly used in various fields like physics, engineering, computer science, and machine learning. It provides a way to multiply two vectors and get a single scalar value, which can give insights into the relationship between the vectors, such as their alignment (considers how much the vectors point in the same or opposite directions) or orthogonality (means perpendicular, they meet at a right angle)."
          },
        ]
      },
      { id: 6,
        name: "Two Rules", 
        value: "",
        image: "src/assets/chapter_one/matmul_rules.jpeg",
        content: [
          { id: 1, 
            title: "",
            value: "For two matrices to be multiplied, the number of columns in the first matrix A must be equal to the number of rows in the second matrix B. If A is a 2 x 3 matrix and B is a 3 x p matrix, the resulting matrix C will be a 2 x p matrix."
          },
        ]
      },
      { id: 7,
        name: "How it Works", 
        value: "",
        image: "src/assets/chapter_one/matmul_work.jpeg",
        content: [
          { id: 1, 
            title: "",
            value: "The element in the ith row and jth column of the resulting matrix C is obtained by taking the dot product of the ith row of matrix A and the jth column of matrix B."
          },
        ]
      },
      { id: 8,
        name: "Applications", 
        value: "Let's talk about what can we do in machine learning with dot production operations.",
        image: "src/assets/chapter_one/dot_attention.jpeg",
        content: [
          { id: 1, 
            title: "Measuring Similarity",
            value: "In machine learning, the dot product is used to measure the similarity between two vectors. For example, in recommendation systems, the similarity between user preferences and item characteristics can be determined using the dot product."
          },
          { id: 2, 
            title: "Optimization",
            value: "In optimization and deep learning, the dot product is often involved in operations within loss functions, gradient calculations, and more."
          },
        ]
      },
      { id: 9,
        name: "4.5 Webs reference", 
        value: "If you wanna know more details about Matrix multiplication, there are 2 webs for reference. It's pretty useful for your understanding about Matrix Multiplication and Dot Product.",
        image: "src/assets/chapter_one/dot_webs.jpeg",
        content: [
          { id: 1, 
            title: "mathisfun.com",
            value: "Math is Fun is a great website for learning and visualizing mathematical concepts in a clear and engaging way. If you're looking to understand matrix multiplication and the dot product, the site offers simple explanations, examples, and interactive tools that can help."
          },
          { id: 2, 
            title: "matrixmultiplication.xyz",
            value: "It is a specialized tool focused on helping users understand and perform matrix multiplication interactively."
          },
        ]
      },
    ]
  },
  { id: 5, 
    name: "TENSOR Manipulation",
    description: "",
    sections: [
      { id: 0,
        name: "5. TENSOR Manipulation", 
        value: "",
        image: "src/assets/chapter_one/manipulation.jpeg",
        content: [
          { id: 1, 
            title: "",
            value: "Tensor manipulation refers to a set of operations and techniques used to modify, reshape, and interact with tensors, which are multi-dimensional arrays (like vectors, matrices, and beyond). In deep learning and scientific computing frameworks like PyTorch and TensorFlow, tensor manipulation is crucial for preparing data, building models, and performing computations."
          },
        ]
      },
      { id: 1,
        name: "5.1 Aggregate Tensor", 
        value: "Aggregating tensors refers to the process of reducing a tensor along one or more dimensions to produce a tensor with a lower rank. Aggregation operations are commonly used in deep learning and other numerical computing tasks to summarize or compress the information contained in a tensor.",
        image: "src/assets/chapter_one/aggregate.jpeg",
        content: [
          { id: 1, 
            title: "max",
            value: "Find the maximum value among all elements or along a specific axis."
          },
          { id: 2, 
            title: "min",
            value: "Find the minimum value among all elements or along a specific axis."
          },
          { id: 3, 
            title: "sum",
            value: "Sum of all elements or sub elements along a specific axis."
          },
          { id: 4, 
            title: "mean",
            value: "Calculates the average of all elements or averages elements along a specific axis."
          },
          { id: 5, 
            title: "argmax",
            value: "Return the indices of the maximum values along a specific axis."
          },
          { id: 6, 
            title: "argmin",
            value: "Return the indices of the minimum values along a specific axis."
          },
        ]
      },
      { id: 2,
        name: "5.2 Chage Data Type, Reshape/Squeeze,/Unsqueeze/Permute Tensor", 
        value: "",
        image: "src/assets/chapter_one/dtype_reshape.jpeg",
        content: [
          { id: 1, 
            title: "Chage the Data Type of Tensor",
            value: "Changing the data type of a tensor is a common operation in PyTorch. You may need to change the data type of a tensor for compatibility with certain operations or to optimize memory usage. Here we use type() to change the target data type of tensor. Changing the data type of tensors allows for greater control over computational efficiency and accuracy in your machine learning models."
          },
          { id: 2, 
            title: "Reshape a Tensor",
            value: "Reshaping a tensor in PyTorch is often necessary when preparing data for different types of operations, especially when working with neural networks, where the input data often needs to be in a specific shape. Here we use reshape() to reshape a target tensor. We're gonna use it quite often, like when we prepare data for neural network, neural networks expect input data to be in a specific shape."
          },
          { id: 3, 
            title: "Squeeze a Tensor",
            value: "In PyTorch, squeeze is a common operations used to modify the shape of tensors by removing dimensions of size 1."
          },
          { id: 4, 
            title: "Unsqueeze a Tensor",
            value: "Just as Squeeze method, Unsqueeze is also a common operation used to modify the shape of tensor by adding dimensions of size 1"
          },
          { id: 5, 
            title: "Permute a Tensor",
            value: "Permuting a tensor in PyTorch refers to rearranging the dimensions of the tensor according to a specified order. This operation is particularly useful when you need to change the order of dimensions for operations like reshaping, aligning data with model requirements, or simply reordering data dimensions."
          },
        ]
      },
      { id: 3,
        name: "5.3 Concatenate/Stack Tensors", 
        value: "",
        image: "src/assets/chapter_one/concat_stack.jpeg",
        content: [
          { id: 1, 
            title: "Concatenate Tensors",
            value: "Concatenating tensors in PyTorch involves joining two or more tensors along an existing dimension. This is useful when you want to merge datasets, combine outputs from different models, or manipulate data in various ways."
          },
          { id: 2, 
            title: "Stack Tensors",
            value: "Tensor stacking is a technique used in PyTorch to combine multiple tensors along a new dimension. It is useful when you want to group a list of tensors into a single tensor while adding an extra dimension to represent this grouping."
          },
        ]
      },
    ]
  },
  { id: 6, 
    name: "TENSOR Transpose",
    description: "",
    sections: [
      { id: 0,
        name: "6. TENSOR Transpose", 
        value: "Transposing a tensor in PyTorch involves swapping two of its dimensions. This operation is commonly used in mathematical computations, especially in linear algebra, where you need to switch the rows and columns of matrices (which are 2D tensors).",
        image: "src/assets/chapter_one/transpose.jpeg",
        content: [
          { id: 1, 
            title: "",
            value: "And this operation is not limited to 2D tensors, you can transpose any 2 dimensions in tensors with more dimensions. Like in a 2D tensors, you can use it to swap any pair of dimensions."
          },
          { id: 2, 
            title: "",
            value: "There are 2 methods you can use to transpose a tensor, the first one is “torch.transpose(dim0, dim1)” function, which return a tensor with dim0 and dim1 swapped. And dim0, dim1 are the dimensions that you wanna transpose. The 2nd function is “tensor.T”, it's only works for 2D tensors and swaps the rows and columns."
          },
          { id: 3, 
            title: "",
            value: "So when to use transpose operations? The one scenario is Matrix Operations, transposing is essential in operations like matrix multiplication, where aline the dimensions correctly in crucial."
          },
          { id: 4, 
            title: "",
            value: "In summary, transposing tensors is a fundamental operation in PyTorch that allows you to swap the positions of two dimensions, making it an essential tool for mathematical operations, data manipulation, and preparing tensors for model input."
          },
        ]
      },
    ]
  },
  { id: 7, 
    name: "7. TENSOR Indexing",
    description: "Tensor indexing in PyTorch refers to accessing specific elements, rows, columns, or subarrays within a tensor using various indexing techniques. This is similar to how you would index lists or arrays in Python but is more powerful and flexible due to PyTorch's support for multi-dimensional tensors.",
    sections: [
      { id: 1,
        name: "7.1 Basic Indexing", 
        value: "",
        image: "src/assets/chapter_one/indexing_one.jpeg",
        content: [
          { id: 1, 
            title: "Single Element",
            value: "You can access single element of a tensor using indices."
          },
          { id: 2, 
            title: "Row or Column",
            value: "You can access entire rows or columns by specifying the index for one dimension and using “:” for the other."
          },
        ]
      },
      { id: 2,
        name: "7.2 Slicing and Boolean Indexing", 
        value: "",
        image: "src/assets/chapter_one/indexing_two.jpeg",
        content: [
          { id: 1, 
            title: "Slicing Indexing",
            value: "You can extract a range of elements from a tensor using slicing, and you can also specify a step value in the slice."
          },
          { id: 2, 
            title: "Boolean Indexing",
            value: "You can use a condition to index a tensor, which will return elements that meet the condition."
          },
        ]
      },
    ]
  },
  { id: 8, 
    name: "TENSOR vs. NumPy Array",
    description: "",
    sections: [
      { id: 1,
        name: "8. TENSOR vs. NumPy Array", 
        value: "PyTorch tensors can easily interact with NumPy arrays, allowing you to convert between the two data types seamlessly. This interoperability is useful when you want to leverage the strengths of both PyTorch and NumPy in your workflow. Here's how you can implement this interaction",
        image: "src/assets/chapter_one/tensor_numpy.jpeg",
        content: [
          { id: 1, 
            title: "PyTorch Tensor -> NumPy Array",
            value: "You can convert a PyTorch tensor to a NumPy array using the .numpy() method. This method creates a view of the tensor data, meaning the NumPy array and the PyTorch tensor share the same underlying data. Any changes made to one will affect the other."
          },
          { id: 2, 
            title: "NumPy Array -> PyTorch Tensor",
            value: "To convert a NumPy array to a PyTorch tensor, you can use the torch.from_numpy() function. Like the .numpy() method, this conversion creates a tensor that shares the same data with the NumPy array"
          },
          { id: 3, 
            title: "Summary",
            value: "Use .numpy() to convert a PyTorch tensor to a NumPy array and torch.from_numpy() to go the other way. Conversions share the same data, so changes in one will reflect in the other. Move tensors to the CPU before converting to NumPy if they're on a GPU."
          },
        ]
      },
    ]
  },
  { id: 9,
    name: "TENSOR Reproducibility",
    description: "",
    sections: [
      { id: 0, 
        name: "9. TENSOR Reproducibility",
        value: "You can get the Same Results on your computer (different platform, PyTorch version, CPU/GPUs) while running the Same Code as I get on mine.",
        image: "",
        content: [
          { id: 1, 
            title: "",
            value: "Tensor reproducibility refers to the ability to consistently produce the same results when performing operations involving tensors in PyTorch, even across different runs of the same code. This is important in machine learning, especially during model training, where you want to ensure that the results are reliable and not due to random variations."
          }, 
          { id: 2, 
            title: "",
            value: "Machine learning algorithms often involve some degree of randomness. For example, when you initialize weights in a neural network, split data into training and test sets, or shuffle data during training, these processes typically involve generating random numbers."
          },
          { id: 3, 
            title: "",
            value: "So how can we achieve tensor reproducibility in PyTorch. The most common approach is to set random seeds."
          }
        ]
      }, 
      { id: 1, 
        name: "9.1 Randomness",
        value: "A real Randomness that each possible outcome is equally likely, and there is no way to predict the next result based on previous ones.",
        image: "", 
        content: [
          { id: 1, 
            title: "", 
            value: "Randomness refers to the lack of any predictable pattern or order in a sequence of events or outcomes. In a truly random process, each possible outcome is equally likely, and there is no way to predict the next result based on previous ones."
          }, 
          { id: 2, 
            title: "",
            value: "For example, flipping a fair coin results in a 50% chance of landing on heads or tails, but each individual flip's result is inherently unpredictable."
          }
        ]
      },
      { id: 2, 
        name: "Randomness in PyTorch",
        value: "",
        image: "src/assets/chapter_one/randpy.jpeg", 
        content: [
          { id: 1, 
            title: "", 
            value: "In PyTorch, randomness is a fundamental part of many operations, and functions like torch.rand() are used to introduce random elements into tensors."
          }, 
          { id: 2, 
            title: "", 
            value: "torch.rand() is a PyTorch function that generates a tensor filled with random numbers drawn from a uniform distribution on the interval [0, 1)."
          }, 
          { id: 3, 
            title: "", 
            value: "Understanding how these functions work and how to control randomness in PyTorch is important for tasks like initializing model parameters, augmenting data, or performing stochastic operations."
          }, 
        ]
      },
      { id: 3, 
        name: "Randomness Features",
        value: "",
        image: "src/assets/chapter_one/randfeat.jpeg", 
        content: [
          { id: 1, 
            title: "Unpredictability", 
            value: "In a random process, future outcomes cannot be determined based on past events. For example, when flipping a fair coin, each flip is independent, and there's a 50% chance of getting heads or tails each time, regardless of previous results."
          }, 
          { id: 2, 
            title: "True Randomness",
            value: "True randomness is typically derived from natural processes, like radioactive decay, thermal noise, or other phenomena that are inherently unpredictable."
          },
          { id: 3, 
            title: "Randomness vs. Pseudorandomness",
            value: "Unlike pseudorandomness, which is generated by algorithms and can be reproduced if the seed is known, true randomness has no underlying deterministic pattern and cannot be precisely reproduced."
          }
        ]
      },
      { id: 4, 
        name: "9.2 PSEUDORANDOMNESS",
        value: "There is no real Randomness, cus the Randomness is Simulated by Computer, is designed, so each step is predictable.",
        image: "", 
        content: [
          { id: 1, 
            title: "", 
            value: "Pseudorandomness refers to sequences of numbers or data that appear random but are generated by a deterministic process. These sequences are created using algorithms known as pseudorandom number generators (PRNGs), which rely on an initial value called a seed."
          }, 
        ]
      },
      { id: 5, 
        name: "PSEUDORANDOMNESS Features",
        value: "",
        image: "src/assets/chapter_one/pseudorandf.jpeg", 
        content: [
          { id: 1, 
            title: "Deterministic", 
            value: "Although pseudorandom sequences look random, they are produced by a specific, repeatable algorithm. If you know the seed and the algorithm, you can predict or replicate the sequence exactly."
          }, 
          { id: 2, 
            title: "Reproducibility", 
            value: "Pseudorandomness allows for reproducibility in computational tasks. By using the same seed, you can generate the same sequence of “random” numbers, which is crucial for debugging, testing, and scientific research."
          }, 
          { id: 3, 
            title: "Not Truly Random", 
            value: "Since pseudorandom numbers are generated by an algorithm, they are not truly random in the sense that they have an underlying pattern or structure, even if that pattern is not immediately apparent. True randomness, on the other hand, would be completely unpredictable and without any deterministic pattern."
          }, 
        ]
      },
      { id: 6, 
        name: "9.3 RANDOM SEED",
        value: "For Performing Repeatable Experiments, to Reduce the Randomness in Neural Network of PyTorch.",
        image: "", 
        content: [
          { id: 1, 
            title: "", 
            value: "So how can we achieve tensor reproducibility in PyTorch. The most common approach is to set random seeds."
          }, 
          { id: 2, 
            title: "", 
            value: "To ensure that the random number generators produce the same sequence of numbers each time you run the code, you should set the seed for PyTorch, NumPy, and Python's random module."
          }, 
          { id: 3, 
            title: "", 
            value: "Setting a random seed ensures that these random processes produce the same results every time you run your code. This is essential for debugging, sharing your work with others, or simply verifying that your model behaves consistently."
          }, 
        ]
      },
      { id: 7, 
        name: "RANDOM SEED in PyTorch",
        value: "",
        image: "src/assets/chapter_one/randseedpy.jpeg", 
        content: [
          { id: 1, 
            title: "Set Random Seeds to 42", 
            value: "The number 42 was popularized by Douglas Adams' science fiction series The Hitchhiker's Guide to the Galaxy, where it is humorously described as “the Answer to the Ultimate Question of Life, the Universe, and Everything.” This has made 42 a geeky, tongue-in-cheek choice among programmers."
          }, 
          { id: 2, 
            title: "42 - No Special Significance", 
            value: "Mathematically, 42 is just like any other number when used as a seed. It doesn't have any inherent properties that make it better or worse for generating random numbers. However, its quirky cultural reference has made it a fun and popular choice."
          }, 
          { id: 3, 
            title: "Can You Use Other Numbers?", 
            value: "Absolutely! You can set the seed to any integer. The choice of 42 is purely conventional and has no impact on the quality of the random numbers generated. What matters is that the seed is set, ensuring reproducibility, not the specific number used."
          }, 
        ]
      },
    ]
  },
  { id: 10, 
    name: "TENSOR on GPU",
    description: "", 
    sections: [
      { id: 0, 
        name: "10. Run TENSOR on GPU",
        value: "Run tensor on GPU, which can significantly speed up computation, especially for large tensors and deep learning models.",
        image: "", 
        content: [
          { id: 1, 
            title: "", 
            value: "Running tensors on GPUs in PyTorch involves moving the tensor data to the GPU and then performing operations on it using GPU acceleration. This can significantly speed up computation, especially for large tensors and deep learning models."
          }
        ]
      }, 
      { id: 1, 
        name: "10.1 GPU",
        value: "GPU short from Graphics Processing Unit, is pretty faster on Tensor Computation as compared to the CPU.",
        image: "",
        content: [
          { id: 1, 
            title: "", 
            value: "A GPU, or Graphics Processing Unit, is a specialized type of processor designed primarily for rendering graphics and performing complex calculations in parallel. Originally developed to handle the heavy computational demands of rendering images and video for gaming and multimedia, GPUs have since become essential in many other fields, including scientific computing, artificial intelligence (AI), and machine learning."
          }
        ]
      },
      { id: 2, 
        name: "GPU Features",
        value: "A GPU, or Graphics Processing Unit, is a specialized type of processor designed primarily for rendering graphics and performing complex calculations in parallel. Originally developed to handle the heavy computational demands of rendering images and video for gaming and multimedia, GPUs have since become essential in many other fields, including scientific computing, artificial intelligence (AI), and machine learning.",
        image: "src/assets/chapter_one/gpuf.jpeg", 
        content: [
          { id: 1, 
            title: "Parallel Processing", 
            value: "GPUs are designed to handle thousands of tasks simultaneously, making them ideal for operations that can be parallelized, such as image processing, simulations, and neural network training.",
          }, 
          { id: 2, 
            title: "Core Structure",
            value: "Unlike a CPU (Central Processing Unit), which typically has a few powerful cores optimized for sequential processing, a GPU has thousands of smaller cores that can work together on multiple tasks at once. This parallelism makes GPUs much faster than CPUs for certain types of computations.",
          },
          { id: 3, 
            title: "Energy Efficiency",
            value: "While GPUs are powerful, they also consume a lot of power, especially when running intensive tasks like gaming or AI model training. However, their ability to perform massive parallel computations makes them more energy-efficient than CPUs for certain tasks.",
          },
          { id: 4, 
            title: "Applications",
            value: "GPUs render high-quality graphics in real-time, making them crucial for modern video games. GPUs accelerate the training of neural networks by processing large datasets in parallel, significantly speeding up the learning process. GPUs are used in simulations, molecular modeling, weather forecasting, and other tasks that require handling large amounts of data quickly.",
          },
        ]
      }, 
      { id: 3, 
        name: "10.2 CUDA",
        value: "CUDA is A Parallel Computing Platform and API that allows Software to Use Certain Type of GPUs, created by NVIDIA.",
        image: "src/assets/chapter_one/cuda.jpg", 
        content: [
          { id: 1, 
            title: "", 
            value: "CUDA, or Compute Unified Device Architecture, is a parallel computing platform and programming model developed by NVIDIA. It allows developers to harness the power of NVIDIA GPUs (Graphics Processing Units) for general-purpose computing, going beyond their traditional use in graphics rendering."
          }
        ]
      },
      { id: 4, 
        name: "CUDA Features",
        value: "", 
        image: "src/assets/chapter_one/cudaf.jpeg", 
        content: [
          { id: 1, 
            title: "Parallel Computing Framework", 
            value: "CUDA enables developers to write programs that can run on NVIDIA GPUs, taking advantage of their massive parallel processing capabilities. This makes it possible to perform computations much faster than on a CPU, especially for tasks that can be parallelized, such as matrix operations, simulations, and neural network training."
          },
          { id: 2, 
            title: "Programming Model", 
            value: "CUDA extends standard programming languages like C, C++, and Python with the addition of keywords and functions that allow for parallel execution on the GPU. Developers can write code that runs on both the CPU (host) and the GPU (device), with the GPU handling the parallelizable parts of the computation.",
          }, 
          { id: 3, 
            title: "Architecture",
            value: "In CUDA, the GPU is viewed as a compute device that operates as a coprocessor to the main CPU. A typical CUDA program involves dividing a problem into smaller subproblems that can be solved concurrently by thousands of lightweight threads running on the GPU cores."
          },
          { id: 4, 
            title: "Benefits",
            value: "CUDA significantly speeds up computation-intensive tasks by leveraging the parallel nature of GPUs. CUDA programs can scale across multiple GPUs, allowing for even greater computational power. CUDA is supported by a rich ecosystem of tools, libraries, and frameworks, including cuDNN (for deep learning) and cuBLAS (for linear algebra), which further simplify development."
          },
        ]
      }, 
      { id: 5, 
        name: "10.3 How to get GPU?",
        value: "",
        image: "src/assets/chapter_one/cudaget.jpeg", 
        content: [
          { id: 1, 
            title: "Check if Your Local Machine Has a GPU", 
            value: "If you have a modern computer, especially a gaming or workstation PC, you might already have a GPU that you can use. If you don't know how to access it, just ask chatGPT or do some Internet survey, you'll get more details. And when you have a GPU, make sure to install the necessary drivers and CUDA toolkit (for NVIDIA GPUs)."
          }, 
          { id: 2, 
            title: "Buy a GPU", 
            value: "If your machine doesn't have a GPU, you can purchase one. The right GPU for you depends on your budget and the tasks you plan to perform."
          }, 
          { id: 3, 
            title: "Use Cloud Services", 
            value: "If you don't have a local GPU or need more powerful GPUs, cloud services offer flexible options, like Amazon Web Services(AWS), NVIDIA GPU Cloud(NGC), Azure - a Microsoft's cloud platform and so on. Here we'll use Google Colab, which you can free to use with access to GPUs (like NVIDIA K80, T4, P100), and you can upgrade to Colab Pro for faster GPUs and longer runtimes. "
          }
        ]
      }, 
      { id: 6, 
        name: "10.4 How to run TENSOR on GPUs?", 
        value: "Running tensors on GPUs in PyTorch involves moving the tensor data to the GPU and then performing operations on it using GPU acceleration. This can significantly speed up computation, especially for large tensors and deep learning models.", 
        image: "src/assets/chapter_one/tensorrun.jpeg", 
        content: [
          { id: 1, 
            title: "Check if GPU is Available", 
            value: "Before moving a tensor to the GPU, you need to check if a GPU is available on your machine."
          },
          { id: 2, 
            title: "Set Device-agnostic code", 
            value: "Device-agnostic code in PyTorch ensures that your code can run seamlessly on both CPUs and GPUs without needing to modify the code. This is particularly useful when you want your code to be flexible and able to utilize GPU acceleration if available, but still run on a CPU if a GPU isn’t present."
          },
          { id: 3, 
            title: "Moving a Tensor to the GPU", 
            value: "Once you’ve confirmed that a GPU is available, you can move a tensor to the GPU using the .to(device) method."
          },
          { id: 4, 
            title: "Moving Tensors Back to CPU", 
            value: "If you need to move the tensor back to the CPU (for example, before converting it to a NumPy array)."
          },
        ]
      }
    ]
  }
]