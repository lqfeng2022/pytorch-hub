export default [
  { id: 1, 
    name: "TENSORs ?",
    description: "",
    sections: [
      { id: 0,
        name: "What's a Tensor", 
        value: "A TENSOR is a multi-dimensional array with a single data type.",
        image: "",
        content: [
          { id: 1, 
            title: "",
            value: "A tensor is a mathematical object that extends the idea of scalars, vectors, and matrices to higher dimensions."
          },
          { id: 2, 
            title: "",
            value: "Tensors are key in fields like physics, engineering, and machine learning, because they let us handle complex, multi-dimensional data in an organized way."
          },
        ]
      },
      { id: 1,
        name: ":: scalar, vector, MATRIX, TENSOR", 
        value: "",
        image: "src/assets/chapter_one/tensors.jpeg",
        content: [
          { id: 1, 
            title: "scalar",
            value: "A scalar is just a single number (a 0-dimensional tensor), we can think of it as a point or a pixel. And we usually write scalars with lowercase."
          },
          { id: 2, 
            title: "vector",
            value: "A Vector is a list of numbers (a 1-dimensional tensor), imaging it as a straight line. As a common practice, we also use lowercase for vectors and bold it."
          },
          { id: 3, 
            title: "MATRIX",
            value: "A Matrix is a grid of numbers (a 2-dimensional tensor), you can treat it as a square or rectangle. In our example, there are 3 rows and each row own 3 column of number. We usually write matrices in uppercase."
          },
          { id: 4, 
            title: "TENSOR",
            value: "Tensors can go beyond two dimensions. For example, a 3D tensor might represent a block of data, like a stack of matrices, and a 4D tensor could represent something that changes over time, like a video. We typically write tensors in uppercase and bold them."
          },
        ]
      },
    ]
  },
  { id: 2, 
    name: "Tensor Creating",
    description: "",
    sections: [
      { id: 0,
        name: "2. Tensor Creating", 
        value: "",
        image: "src/assets/chapter_one/tensorcreate.jpeg",
        content: [
          { id: 1, 
            title: "",
            value: "A random tensor is simply a tensor filled with random numbers. These numbers usually come from a specific probability distribution, like a Normal Distribution—where most numbers cluster around the mean, or a Uniform Distribution—where all numbers within a certain range have an equal chance of being selected."
          },
        ]
      },
      { id: 1,
        name: "2.1 Random Tensor", 
        value: "",
        image: "src/assets/chapter_one/randomtensor.jpeg",
        content: [
          { id: 1, 
            title: "What's a random tensor?",
            value: "A random tensor is simply a tensor filled with random numbers. These numbers usually come from a specific probability distribution, like a Normal Distribution—where most numbers cluster around the mean, or a Uniform Distribution—where all numbers within a certain range have an equal chance of being selected."
          },
          { id: 2, 
            title: "Why use random tensor?",
            value: "Random tensors are often used in machine learning and deep learning for various purpose, such as initializing Weights. When you're setting up a neural network, the weights of the network's layers need to start with some initial values, and they are typically small random numbers. Starting with random values helps kick off the learning process in a way that prevents symmetry, allowing the network to learn effectively."
          },
          { id: 3, 
            title: "How to create a random tensor?",
            value: "In PyTorch, you can create a random tensor using functions like torch.rand() for uniform distributions, torch.randn() for normal distributions, and others, depending on your needs."
          },
        ]
      },
      { id: 2,
        name: "2.2 Zeros Tensor & Ones Tensor", 
        value: "",
        image: "src/assets/chapter_one/zerosOnes.jpeg",
        content: [
          { id: 1, 
            title: "What are zeros ones tensors?",
            value: "A zeros or ones tensor is a tensor where every element is initialized to zero or one. It's like an array filled with zeros or ones but it can be multi-dimensional and more complex."
          },
          { id: 2, 
            title: "Why use zeros and ones tensors?",
            value: "In neural networks, bias terms (parameters) are often initialized to zero, because this doesn't bring any asymmetry or preference during training. Sometimes you might want to initialize certain parameters or tensors to one instead of zero, especially when dealing with multiplicative operations. There are many other reasons, and we'll cover them as thet came up in specific scenarios."
          },
          { id: 3, 
            title: "How to create zeros and ones tensors?",
            value: "You can create zeros and ones tensors in PyTorch with torch.zeros() and torch.ones(). For example, torch.zeros(3, 3) and torch.ones(3, 3) will generate two 3x3 matrices filled with zeros and ones, respectively."
          },
        ]
      },
      { id: 3,
        name: "2.3 Range Tensor", 
        value: "",
        image: "src/assets/chapter_one/rangetensor.jpeg",
        content: [
          { id: 1, 
            title: "What's a range tensor?",
            value: "A range tensor is a tensor that contains a sequence of numbers within a specified range and step size. It's similar to Python's built-in range() function, but instead of generating a list, it generates a tensor."
          },
          { id: 2, 
            title: "Why use a range tensor?",
            value: "Range tensors are often used to create index values when slicing or performing operations on specific ranges within other tensors.."
          },
          { id: 3, 
            title: "How to create a range tensor?",
            value: "In PyTorch, you can create a range tensor using functions like torch.arange(start, end, step). And the start means the starting value of the sequence (inclusive), end means the ending value of the sequence (exclusive), and step means the difference between each consecutive value."
          },
        ]
      },
      { id: 4,
        name: "2.4 Tensor Like", 
        value: "",
        image: "src/assets/chapter_one/tensorlike.jpeg",
        content: [
          { id: 1, 
            title: "",
            value: "When we talk about “tensor like” in PyTorch, we're usually referring to functions like torch.zeros_like() and torch.ones_like(). These functions are used to create new tensors that have the same shape and device as a given tensor but are filled with zeros or ones."
          },
        ]
      },
    ]
  },
  { id: 3, 
    name: "Tensor Attributes",
    description: "",
    sections: [
      { id: 0,
        name: "3. Tensor Attributes", 
        value: "In PyTorch, a tensor comes with several important attributes that give you information about their structure, data type, and where they are stored (CPU or GPU). The main attributes you'll often work with are 'shape', 'ndim', 'dtype', and 'device'. Here's a brief overview of each:",
        image: "src/assets/chapter_one/attributes.jpeg",
        content: [
          { id: 1, 
            title: "shape",
            value: "The 'shape' attribute tells you the size of each dimension of a tensor. It's a tuple of integers where each integer represents the size of the tensor along a specific dimension. Knowing the shape of a tensor is crucial because it defines the structure of the data you're working with. It tells you how many elements are in each dimension."
          },
          { id: 2, 
            title: "ndim",
            value: "The 'ndim' attribute gives you the number of dimensions (axes) of a tensor. Understanding the number of dimensions is essential for operations like reshaping, and when you're visualizing the structure of data."
          },
          { id: 3, 
            title: "dtype",
            value: "The 'dtype' attribute indicates the data type of the elements stored in a tensor, like 'torch.float32', 'torch.int64', etc. The data type determines the precision and range of the numbers in a tensor. Choosing the right data type can affect both performance and memory usage."
          },
          { id: 4, 
            title: "device",
            value: "The 'device' attribute tells you where the tensor's data is stored: either on the CPU or a GPU. Tensors on a GPU will have their device set to something like 'cuda:0'. Knowing where a tensor is stored is crucial when you're working with GPUs for accelerated computation. You need to ensure that all tensors involved in operations are 'on the same device' to avoid errors."
          },
        ]
      },
      { id: 1,
        name: "3.1 the SHAPE", 
        value: "Working with the shape attribute in PyTorch can sometimes be tricky, especially when you're manipulating tensors or performing operations that involve reshaping, matrix multiplication. To help visualize and understand the shape, let's look at an example.",
        image: "src/assets/chapter_one/shape.jpeg",
        content: [
          { id: 1, 
            title: "",
            value: "In this example, we have a tensor X created using 'torch.rand(3, 4)'. The shape attribute of this tensor is 'torch.Size([3, 4])'. The first number (highlighted in pink) represents the outer dimension (3), which corresponds to the number of rows. The second number (highlighted in green) represents the inner dimension (4), which corresponds to the number of columns. So, the tensor is structured as a 3x4 grid, where each cell contains a random number."
          },
          { id: 2, 
            title: "",
            value: "This pattern, moving from the outer dimension to the inner dimension, is like going from rows to columns in a grid."
          },
          { id: 3, 
            title: "",
            value: "The ndim attribute tells us the number of dimensions the tensor has. In this case, 'X.ndim' is 2, indicating that this is a 2-dimensional tensor, which fits the grid-like structure."
          },
          { id: 4, 
            title: "",
            value: "By default, the data type (dtype) of the tensor is torch.float32, and the tensor is stored on the CPU (device)."
          },
        ]
      },
    ]
  },
  { id: 4, 
    name: "Tensor Operations",
    description: "",
    sections: [
      { id: 0,
        name: "4. Tensor Operations", 
        value: "",
        image: "src/assets/chapter_one/operations.jpeg",
        content: [
          { id: 1, 
            title: "",
            value: "In the world of machine learning, tensors are the building blocks for representing complex data. Tensor operations allow us to manipulate this data efficiently, enabling everything from basic arithmetic to advanced transformations. Just as we perform operations on scalars, vectors, and matrices, we can apply similar operations to tensors, making them essential for tasks like reshaping data, performing matrix multiplications, and more."
          },
        ]
      },
      { id: 1,
        name: "4.1 Basic Operations", 
        value: "",
        image: "src/assets/chapter_one/addsub.jpeg",
        content: [
          { id: 1, 
            title: "",
            value: "You can add or subtract a number from a tensor, which applies the operation to each element of the tensor. Additionally, you can add or subtract two tensors of the same shape by performing the operation element-wise."
          },
          { id: 2, 
            title: "",
            value: "Similar to addition and subtraction, you can multiply or divide each element of a tensor by a number or perform element-wise multiplication or division with another tensor of the same shape."
          },
        ]
      },
      { id: 2,
        name: "4.2 Matrix Multiplication", 
        value: "",
        image: "src/assets/chapter_one/matmul.jpeg",
        content: [
          { id: 1, 
            title: "",
            value: "Matrix multiplication is a fundamental operation in linear algebra where two matrices are multiplied to produce a third matrix. This operation is crucial in various fields, including computer science, engineering, physics, and especially in machine learning, where it plays a key role in tasks such as training neural networks, data transformations, and efficiently handling large datasets."
          },
          { id: 2, 
            title: "",
            value: "Matrix multiplication, also known as the dot product when applied to vectors, involves taking the sum of the products of corresponding elements from a row of the first matrix and a column of the second matrix. For matrix multiplication to be valid, the number of columns in the first matrix must be equal to the number of rows in the second matrix."
          },
        ]
      },
      { id: 3,
        name: ":: How it Works", 
        value: "",
        image: "src/assets/chapter_one/matmul_work.jpeg",
        content: [
          { id: 1, 
            title: "",
            value: "To multiply matrices A and B, we compute the dot product of each row of A with each column of B. The resulting matrix C has a shape of [2, 2]. Each element of C is obtained by dotting a row of A with a corresponding column of B: the first row of A with the first and second columns of B for the first row of C, and the second row of A with the first and second columns of B for the second row of C."
          },
        ]
      },
      { id: 4,
        name: ":: Two Rules", 
        value: "",
        image: "src/assets/chapter_one/matmul_rules.jpeg",
        content: [
          { id: 1, 
            title: "",
            value: "For two matrices to be multiplied, the number of columns in the first matrix (A) must equal the number of rows in the second matrix (B). If A is a 2 x 3 matrix and B is a 3 x p matrix, the resulting matrix (C) will have a shape of 2 x p."
          },
        ]
      },
      { id: 5,
        name: ":: Learning Webs", 
        value: "If you want to dive deeper into matrix multiplication, I recommend checking out these two websites. They offer valuable insights that can enhance your understanding of both matrix multiplication and the dot product.",
        image: "src/assets/chapter_one/dot_webs.jpeg",
        content: [
          { id: 1, 
            title: "mathisfun.com",
            value: "Math is Fun is a great website for learning and visualizing mathematical concepts in a clear and engaging way. If you're looking to understand matrix multiplication and dot product, this site offers simple but clear explanations, examples, and interactive tools that make these concepts easy to grasp."
          },
          { id: 2, 
            title: "matrixmultiplication.xyz",
            value: "It is a specialized tool focused on helping users understand and perform matrix multiplication interactively. It allows you to see each step of the calculation process, making it easier to grasp how matrix multiplication works."
          },
        ]
      },
      { id: 6,
        name: ":: Dot-Product", 
        value: "",
        image: "src/assets/chapter_one/dot.jpeg",
        content: [
          { id: 1, 
            title: "",
            value: "The dot product is a fundamental operation in vector algebra, widely used in fields like physics, engineering, computer science, and machine learning. It multiplies two vectors to produce a single scalar value, providing insights into the relationship between the vectors. The dot product can reveal how aligned the vectors are, indicating whether they point in the same or opposite directions, and can also determine if the vectors are perpendicular, meaning they meet at a right angle."
          },
        ]
      },
      { id: 7,
        name: ":: Dot-Product in Transformer", 
        value: "The dot product is a fundamental operation in the Transformer architecture, enabling the model to weigh the importance of different parts of the input efficiently and effectively, which is why it has become so powerful in tasks like translation, summarization, and more.",
        image: "src/assets/chapter_one/dot_attention.jpeg",
        content: [
          { id: 1, 
            title: "Measuring Similarity",
            value: "In machine learning, the dot product is a crucial operation for measuring the similarity between two vectors. For example, in recommendation systems, the dot product can quantify how closely a user's preferences align with the characteristics of an item. This concept underlies techniques like cosine similarity, which normalizes the dot product to account for vector magnitudes."
          },
          { id: 2, 
            title: "Optimization",
            value: "In optimization and deep learning, the dot product is integral to various operations within loss functions and gradient calculations. It plays a crucial role in backpropagation, where gradients are computed as part of the process to minimize the loss function and update model parameters effectively."
          },
          { id: 3, 
            title: "Feature Extraction and Representation",
            value: "The dot product is also used in feature extraction and representation. For instance, in neural networks, dot products are used within layers to combine input features with weights, producing activations that are then passed through non-linear functions. This operation underpins how networks learn complex patterns in data."
          },
        ]
      },
      { id: 8,
        name: "4.3 Aggregating Operations", 
        value: "Aggregating tensors refers to the process of reducing a tensor along one or more dimensions to produce a tensor with a lower rank. Aggregation is commonly used in deep learning to summarize or compress the information contained in a tensor, making it more manageable for further processing.",
        image: "src/assets/chapter_one/aggregate.jpeg",
        content: [
          { id: 1, 
            title: "max()",
            value: "Find the maximum value among all elements in a tensor or along a specific axis."
          },
          { id: 2, 
            title: "min()",
            value: "Find the minimum value among all elements in a tensor or along a specific axis."
          },
          { id: 3, 
            title: "sum()",
            value: "Calculate the sum of all elements in a tensor or along a specific axis."
          },
          { id: 4, 
            title: "mean()",
            value: "Calculates the average of all elements in a tensor or along a specific axis."
          },
          { id: 5, 
            title: "argmax()",
            value: "Return the indices of the maximum values along a specific axis of a tensor"
          },
          { id: 6, 
            title: "argmin()",
            value: "Return the indices of the minimum values along a specific axis of a tensor."
          },
        ]
      },
    ]
  },
  { id: 5, 
    name: "Tensor Manipulation",
    description: "",
    sections: [
      { id: 0,
        name: "5. Tensor Manipulation", 
        value: "",
        image: "src/assets/chapter_one/manipulation.jpeg",
        content: [
          { id: 1, 
            title: "",
            value: "We've discussed various tensor operations, such as addition, subtraction, dividing, multiplication (element-wise) and matrix multiplication, which are fundamental in deep learning. These operations are part of a broader concept known as tensor manipulation."
          },
          { id: 2, 
            title: "",
            value: "Tensor manipulation encompasses a range of techniques used to aggregate, reshape, squeeze, unsqueeze, and permute tensors, change their data types, and perform interactions such as concatenation and stacking. In deep learning, mastering tensor manipulation is crucial for tasks such as data preparation, model building, and performing computations."
          },
        ]
      },
      { id: 1,
        name: "5.1 Reshaping a Tensor", 
        value: "",
        image: "src/assets/chapter_one/dtype_reshape.jpeg",
        content: [
          { id: 1, 
            title: "Change the Data Type of a Tensor",
            value: "Changing the data type of a tensor is a common operation in PyTorch. You may need to change the data type of a tensor for compatibility with certain operations or to optimize memory usage. Here we use type() to change the target data type of a tensor. Changing the data type of tensors allows for greater control over computational efficiency and accuracy in your machine learning models."
          },
          { id: 2, 
            title: "Reshape a Tensor",
            value: "In PyTorch, reshaping a tensor is often necessary when preparing data for different types of operations. This is super important with neural networks since they usually expect the input data to be in a specific shape. Here we use reshape() to change the shape of target tensor. We'll use it quite often when preparing data for neural networks, which expect input data to be in a specific shape."
          },
          { id: 3, 
            title: "Squeeze a Tensor",
            value: "In PyTorch, squeeze is a common operation used to adjust the shape of tensors by removing dimensions with a size of 1."
          },
          { id: 4, 
            title: "Unsqueeze a Tensor",
            value: "Just as the squeeze method, unsqueeze is a common operation used to modify the shape of a tensor by adding dimensions of size 1"
          },
          { id: 5, 
            title: "Permute a Tensor",
            value: "Permuting a tensor in PyTorch means rearranging its dimensions according to a specified order. This operation is especially useful when you need to change the order of dimensions for tasks like reshaping, aligning data with model requirements, or simply reordering data dimensions."
          },
        ]
      },
      { id: 2,
        name: ":: Transposing a Tensor", 
        value: "Transposing a tensor in PyTorch is all about swapping two of its dimensions. This operation is commonly used in mathematical computations, especially in linear algebra, where you need to switch the rows and columns of matrices. But it's not just for matrices — you can use transposition on any tensor, no matter how many dimensions it has.",
        image: "src/assets/chapter_one/transpose.jpeg",
        content: [
          { id: 1, 
            title: "",
            value: "Transposing isn't just for 2D tensors—you can swap any two dimensions in tensors with more dimensions. Just like with 2D tensors, where you can switch rows and columns, you can use transposition to rearrange any pair of dimensions in higher-dimensional tensors too"
          },
          { id: 2, 
            title: "",
            value: "There are two ways to transpose a tensor in PyTorch. The first method is using the torch.transpose(dim0, dim1) function, which lets you swap any two dimensions of a tensor, making it super flexible for tensors of any size. The second method is the shorthand tensor.T, which is specifically for 2D tensors and simply swaps the rows and columns."
          },
          { id: 3, 
            title: "",
            value: "So when to use transpose operations? The one scenario is Matrix Operations, transposing is essential in operations like matrix multiplication, where aline the dimensions correctly in crucial."
          },
        ]
      },
      { id: 3,
        name: "5.2 Changing Data Type", 
        value: "PyTorch tensors can easily interact with NumPy arrays, making it simple to convert between the two. This seamless interoperability is super useful when you want to take advantage of both PyTorch's capabilities and NumPy's strengths in your workflow. Here's how you can implement this interaction:",
        image: "src/assets/chapter_one/tensor_numpy.jpeg",
        content: [
          { id: 1, 
            title: "PyTorch Tensor -> NumPy Array",
            value: "You can convert a PyTorch tensor to a NumPy array using numpy() method. This method creates a view of the tensor data, meaning the NumPy array and the PyTorch tensor share the same underlying data. So, any changes made to one will affect the other."
          },
          { id: 2, 
            title: "NumPy Array -> PyTorch Tensor",
            value: "To convert a NumPy array to a PyTorch tensor, you can use the torch.from_numpy() function. Like the numpy() method, this conversion creates a tensor that shares the same data with the NumPy array, ensuring any changes in the NumPy array will also reflect in the tensor."
          },
          { id: 3, 
            title: "Summary",
            value: "Use numpy() to convert a PyTorch tensor to a NumPy array and torch.from_numpy() to go the other way. Since both conversions share the same underlying data, so changes in one will reflect in the other.If your tensor is on a GPU, remember to move it to the CPU before converting it to a NumPy array."
          },
        ]
      },
      // Move tensor run on GPUs here,
      // cus that's all about changing the device (cpu/gpu)
      { id: 4,
        name: "5.4 Concatenation and Stacking", 
        value: "",
        image: "src/assets/chapter_one/concat_stack.jpeg",
        content: [
          { id: 1, 
            title: "Concatenate Tensors",
            value: "Concatenating tensors in PyTorch involves joining two or more tensors along a specific dimension. This is useful when you want to merge datasets, combine outputs from different models, or manipulate data in various ways."
          },
          { id: 2, 
            title: "Stack Tensors",
            value: "Tensor stacking in PyTorch is a technique used to combine multiple tensors along a new dimension. This is useful when you want to group a list of tensors into a single tensor, adding an extra dimension to represent this grouping."
          },
        ]
      },
    ]
  },
  { id: 6, 
    name: "Tensor Indexing",
    description: "",
    sections: [
      { id: 0,
        name: "6. Tensor Indexing", 
        value: "Tensor indexing in PyTorch is all about accessing specific elements, rows, columns, or subarrays within a tensor.",
        image: "",
        content: [
          { id: 1, 
            title: "",
            value: "It is similar to how you would index lists or arrays in Python, but it's way more powerful and flexible because PyTorch supports multi-dimentional tensors. Whether you're working with a simple 2D matrix or a more complex tensor, indexing lets you dive into the exact data you need."
          },
        ]
      },
      { id: 1,
        name: "6.1 Basic Indexing", 
        value: "",
        image: "src/assets/chapter_one/indexing_one.jpeg",
        content: [
          { id: 1, 
            title: "Single Element",
            value: "You can access a single element of a tensor using indices. Just like lists in Python, you specify the position within each dimension."
          },
          { id: 2, 
            title: "Row or Column",
            value: "If you want to access an entire row or column, you can do that by specifying the index for one dimension and using “:” for the other. This is a quick way to grab all the elements along a particular axis."
          },
        ]
      },
      { id: 2,
        name: "6.2 Slicing and Boolean Indexing", 
        value: "",
        image: "src/assets/chapter_one/indexing_two.jpeg",
        content: [
          { id: 1, 
            title: "Slicing Indexing",
            value: "You can extract a range of elements from a tensor using slicing, similar to how you would slice a list in Python. You can even specify a step value to skip elements within that range."
          },
          { id: 2, 
            title: "Boolean Indexing",
            value: "You can alse use a condition to index a tensor, which will return all the elements that meet that condition. This is super handy for filtering data based on specific criteria."
          },
        ]
      },
    ]
  },
  { id: 7,
    name: "Tensor Reproducibility",
    description: "",
    sections: [
      { id: 0, 
        name: "7. Tensor Reproducibility",
        value: "You can get the Same Results on your computer - regardless of platform, PyTorch version, or hardware(CPU/GPUs) - as I get on mine, simply running the same code.",
        image: "",
        content: [
          { id: 1, 
            title: "",
            value: "Tensor reproducibility means consistently getting the same results when performing operations with tensors in PyTorch, even when running the same code multiple times. This is crucial in machine learning, particularly during model training, where reliable results are key to ensuring your outcomes aren't due to random variations."
          }, 
          { id: 2, 
            title: "",
            value: "Machine learning algorithms often involve some randomness. For instance, when you initialize neural network weights, split data into training and test sets, or shuffle data during training. These processes typically involve random number generation."
          },
          { id: 3, 
            title: "",
            value: "So to achieve tensor reproducibility in PyTorch, the most common approach is to set random seeds. This helpss control the randomness and ensures consistent results across different runs."
          }
        ]
      }, 
      { id: 1, 
        name: "7.1 Randomness",
        value: "True Randomness means that each possible outcome is equally likely, and there is no way to predict the next result based on previous ones.",
        image: "", 
        content: [
          { id: 1, 
            title: "", 
            value: "Randomness refers to the absence of any predictable pattern or order in a sequence of events or outcomes. In a truly random process, each possible outcome is equally likely."
          }, 
          { id: 2, 
            title: "",
            value: "For instance, when flipping a fair coin, there's a 50% chance of landing on heads or tails, but the outcome of each flip remains unpredictable."
          }
        ]
      },
      { id: 2, 
        name: ":: Randomness in PyTorch",
        value: "",
        image: "src/assets/chapter_one/randpy.jpeg", 
        content: [
          { id: 1, 
            title: "", 
            value: "Randomness plays a crucial role in many PyTorch operations. Functions like torch.rand() are commonly used to introduce random elements into tensors."
          }, 
          { id: 2, 
            title: "", 
            value: "torch.rand() generates a tensor filled with random numbers drawn from a uniform distribution in the interval [0, 1)."
          }, 
          { id: 3, 
            title: "", 
            value: "Understanding how these functions work and how to control randomness in PyTorch is important for tasks such as initializing model parameters, augmenting data, and performing stochastic operations."
          }, 
        ]
      },
      { id: 3, 
        name: ":: Randomness Features",
        value: "",
        image: "src/assets/chapter_one/randfeat.jpeg", 
        content: [
          { id: 1, 
            title: "Unpredictability", 
            value: "In a random process, future outcomes cannot be determined based on past events. For example, when flipping a fair coin, each flip is independent, and there's a 50% chance of getting heads or tails each time, regardless of previous results."
          }, 
          { id: 2, 
            title: "True Randomness",
            value: "True randomness is typically derived from natural processes, like radioactive decay or thermal noise - phenomena that are inherently unpredictable and not influenced by previous events."
          },
          { id: 3, 
            title: "Randomness vs. Pseudorandomness",
            value: "Unlike pseudorandomness, true randomness is generated by algorithms and can be reproduced if the seed is known. While pseudorandomness can simulate randomness, it has an underlying deterministic pattern, unlike true randomness, which cannot be precisely reproduced"
          }
        ]
      },
      { id: 4, 
        name: "7.2 PSEUDORANDOMNESS",
        value: "There is no real randomness in computers because the randomness is simulated. It's designed, so each step is predictable.",
        image: "", 
        content: [
          { id: 1, 
            title: "", 
            value: "Pseudorandomness refers to sequences that appear random but are generated by a deterministic process using pseudorandom number generators (PRNGs). These sequences depend on an initial value known as a seed."
          }, 
        ]
      },
      { id: 5, 
        name: ":: PSEUDORANDOMNESS Features",
        value: "",
        image: "src/assets/chapter_one/pseudorandf.jpeg", 
        content: [
          { id: 1, 
            title: "Deterministic", 
            value: "Pseudorandom sequences may appear random, but they are produced by a specific, repeatable algorithm. If you know the seed and the algorithm, you can predict or replicate the sequence exactly."
          }, 
          { id: 2, 
            title: "Reproducibility", 
            value: "Pseudorandomness allows for reproducibility in computational tasks. By using the same seed, you can generate the same sequence of “random” numbers, which is crucial for debugging, testing, and scientific research."
          }, 
          { id: 3, 
            title: "Not Truly Random", 
            value: "Pseudorandom numbers are generated by an algorithm, so they are not truly random. They have an underlying pattern or structure, even if that pattern is not immediately obvious. True randomness, on the other hand, would be completely unpredictable and lacks any deterministic pattern."
          }, 
        ]
      },
      { id: 6, 
        name: "7.3 RANDOM SEED",
        value: "A random seed is used to perform repeatable experiments by reducing randomness in neural networks within PyTorch.",
        image: "", 
        content: [
          { id: 1, 
            title: "", 
            value: "So how can we achieve tensor reproducibility in PyTorch? The most common approach is to set random seeds."
          }, 
          { id: 2, 
            title: "", 
            value: "To ensure that the random number generators produce the same sequence of numbers each time you run the code, you should set the seed for PyTorch, NumPy, and Python's random module."
          }, 
          { id: 3, 
            title: "", 
            value: "Setting a random seed ensures that these random processes produce the same results every time you run your code. This is essential for debugging, sharing your work with others, or just making sure your model behaves consistently."
          }, 
        ]
      },
      { id: 7, 
        name: ":: RANDOM SEED in PyTorch",
        value: "",
        image: "src/assets/chapter_one/randseedpy.jpeg", 
        content: [
          { id: 1, 
            title: "Set Random Seed to 42", 
            value: "The number 42 is famously known from Douglas Adams' science fiction series 'The Hitchhiker's Guide to the Galaxy', where it's humorously dubbed “the Answer to the Ultimate Question of Life, the Universe, and Everything”. Because of this, 42 has become a geeky, tongue-in-cheek choice among programmers."
          }, 
          { id: 2, 
            title: "42 - No Special Significance", 
            value: "Mathematically, 42 is just another number when used as a seed. It doesn't have any special properties that make it better or worse for generating random numbers. Its popularity comes from its quirky cultural reference, not from any unique mathematical advantage."
          }, 
          { id: 3, 
            title: "Can You Use Other Numbers?", 
            value: "Absolutely! You can set the seed to any integer. The choice of 42 is purely conventional and has no impact on the quality of the random numbers generated. What matters is that the seed is set, ensuring reproducibility, not the specific number used."
          }, 
        ]
      },
    ]
  },
  { id: 8, 
    name: "Tensor on GPUs",
    description: "", 
    sections: [
      { id: 0, 
        name: "8. Tensor on GPUs",
        value: "Running tensors on GPUs can significantly speed up computation, especially when dealing with large tensors or training deep learning models.",
        image: "", 
        content: [
          { id: 1, 
            title: "", 
            value: "Running tensors on GPUs in PyTorch involves transferring the tensor data to GPU and then performing operations on it using GPU acceleration. This can greately speed up computations, particularly when working with large tensors and training deep learning models."
          }
        ]
      }, 
      { id: 1, 
        name: "8.1 Run Tensor on GPUs", 
        value: "To run tensors on GPUs with PyTorch, you need to move your tensor data to the GPU and then perform operations using GPU acceleration. This can really speed up computation, especially when working with large tensors and deep learning models.", 
        image: "src/assets/chapter_one/tensorrun.jpeg", 
        content: [
          { id: 1, 
            title: "Check if GPU is Available", 
            value: "First, you need to check if a GPU is available on your machine before you move a tensor to it."
          },
          { id: 2, 
            title: "Set Device-agnostic code", 
            value: "Write your code in a way that it can run on both CPUs and GPUs without changes. This makes your code flexible, so it can use GPU acceleration if available but still run on a CPU if not."
          },
          { id: 3, 
            title: "Moving a Tensor to the GPU", 
            value: "Once you've confirmed a GPU is available, you can move your tensor to the GPU using the .to(device) method."
          },
          { id: 4, 
            title: "Moving Tensors Back to CPU", 
            value: "If you need to move the tensor back to the CPU (for instance, before converting it to a NumPy array), you can do so easily."
          },
        ]
      },
      { id: 2, 
        name: "8.2 GPUs",
        value: "A GPU, short from Graphics Processing Unit, is much faster at tensor computations compared to a CPU.",
        image: "",
        content: [
          { id: 1, 
            title: "", 
            value: "GPU is a specialized processor designed primarily for rendering graphics and performing complex calculations in parallel. While originally developed to handle the heavy computational demands of gaming and multimedia, GPUs have become essential in fields like scientific computing, artificial intelligence (AI), and machine learning due to their ability to eficiently manage large-scale computations."
          }
        ]
      },
      { id: 3, 
        name: ":: GPU Features",
        value: "",
        image: "src/assets/chapter_one/gpuf.jpeg", 
        content: [
          { id: 1, 
            title: "Parallel Processing", 
            value: "GPUs are designed to handle thousands of tasks at once, making them ideal for operations that can be parallelized, such as image processing, simulations, and training neural network.",
          }, 
          { id: 2, 
            title: "Core Structure",
            value: "Unlike CPUs, which typically have a few powerful cores optimized for sequential processing, GPUs feature thousands of smaller cores that can work on multiple tasks concurrently. This architecture makes GPUs significantly faster than CPUs for certain types of computations.",
          },
          { id: 3, 
            title: "Energy Efficiency",
            value: "While GPUs are powerful, they also consume a lot of power, especially during demanding tasks like gaming or AI model training. However, their ability to handle numerous tasks at once often makes them more energy-efficient than CPUs for these certain applications.",
          },
          { id: 4, 
            title: "Applications",
            value: "GPUs are crucial for rendering high-quality graphics in real-time, which is essential for modern video games. They also accelerate the training of neural networks by processing large datasets in parallel. Additionally, GPUs are used in simulations, molecular modeling, weather forecasting, and other tasks that require a fast processing of large amounts of data.",
          },
        ]
      }, 
      { id: 4, 
        name: "8.3 CUDA",
        value: "",
        image: "src/assets/chapter_one/cuda.jpg", 
        content: [
          { id: 1, 
            title: "", 
            value: "CUDA (Compute Unified Device Architecture) is a parallel computing platform and programming model developed by NVIDIA. It lets developers tap into the power of NVIDIA GPUs for tasks beyound just graphics rendering, making it possible to use GPUs for feneral-purpose computing."
          }
        ]
      },
      { id: 5, 
        name: ":: CUDA Features",
        value: "", 
        image: "src/assets/chapter_one/cudaf.jpeg", 
        content: [
          { id: 1, 
            title: "Parallel Computing Framework", 
            value: "CUDA enables developers to write programs that run on NVIDIA GPUs, taking full advantage of their massive parallel processing power. This means computations can be done much faster than on a CPU, especially for tasks that can be parallelized, like matrix operations, simulations, and neural network training."
          },
          { id: 2, 
            title: "Programming Model", 
            value: "CUDA builds on standard programming languages like C, C++, and Python by adding special keywords and functions for parallel execution on the GPU. This allows developers to write code that runs on both the CPU (host) and the GPU (device), with the GPU handling the parts of the computation that can be run in parallel.",
          }, 
          { id: 3, 
            title: "Architecture",
            value: "In CUDA, the GPU acts as a co-processor to the main CPU. A typical CUDA program breaks a problem into smaller subproblems that can be tackled concurrently by thousands of lightweight threads running on the GPU cores."
          },
          { id: 4, 
            title: "Benefits",
            value: "CUDA greatly speeds up computation-heavy tasks by using the parallel nature of GPUs. It supports scaling across multiple GPUs for even more power. Plus, there's a rich set of tools, libraries, and frameworks, including cuDNN for deep learning and cuBLAS for linear algebra, which make development easier."
          },
        ]
      }, 
      { id: 6, 
        name: "8.4 Get GPUs",
        value: "",
        image: "src/assets/chapter_one/cudaget.jpeg", 
        content: [
          { id: 1, 
            title: "Check if Your Local Machine Has a GPU", 
            value: "If you've got a modern computer, especially a gaming or workstation PC, you might already have a GPU ready to go. If you're not sure how to check, just ask chatGPT or do a quick onlien search for guidance. Once you find out, make sure to install the necessary drivers and the CUDA toolkit if you have an NVIDIA GPU."
          }, 
          { id: 2, 
            title: "Buy a GPU", 
            value: "If your machine doesn't have a GPU, you can buy one. The best GPU for you will depend on your budget and the tasks you want to perform."
          }, 
          { id: 3, 
            title: "Use Cloud Services", 
            value: "If you don't have a local GPU or need more powerful GPUs, cloud services offer flexible options. You can use platforms like Amazon Web Services (AWS), NVIDIA GPU Cloud (NGC), or Microsoft Azure. For a straightforward option, Google Colab offers free access to GPUs like NVIDIA K80, T4, and P100. You can also upgrade to Colab Pro for faster GPUs and longer runtimes."
          }
        ]
      }, 
    ]
  }
]