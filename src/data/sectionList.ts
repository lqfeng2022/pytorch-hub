export default [
  { id: 0, 
    name: "Introduction", 
    link: '/introduction', 
    items: [], 
    description: [] 
  },
  { id: 1,
    name: "0. Artificial Intelligence",
    link: "/artificial-intelligence",
    items: [
      { id: 1, name: "AI - Artificial Intelligence", link: "artificial-intelligence" },
      { id: 2, name: "ML - Machine Learning", link: "machine-learing" },
      { id: 3, name: "DL - Deep Learning", link: "deep-learning"},
      { id: 4, name: "NN - Neural Network", link: "neural-network" },
      { id: 5, name: "Frameworks and Libraries", link: "libraries" },
      { id: 6, name: "PyTorch", link: "pytorch" },
      { id: 7, name: "Prerequisites", link: "prerequisites" },
      { id: 8, name: "How to take this Course", link: "how-to-take-course" },
    ],
    description: [
      { id: 1, 
        title: "",
        value: "In this chapter, we'll dive into some fundamental concepts of Artificial Intelligence and explore a few key AI frameworks and libraries."
      },
      { id: 2, 
        title: "",
        value: "First off, we gotta get what AI, Machine Learning, Deep Learning, and Neural Networks really are. After that, I'll introduce you to some of the most useful Machine Learning frameworks and libraries. For beginners, PyTorch stands out as a powerful and user-friendly framework, so we'll be using PyTorch to build our next projects."
      },
      { id: 3, 
        title: "",
        value: "Before we get started, it's helpful to have a basic understanding of Python, such as knowing about variables, functions, classes, data types, and so on. But don't worry if you're not up to speed on Python yet, I've got another course that covers it, and it's pretty straightforward. I'm confident you'll pick it up quickly."
      },
      { id: 4, 
        title: "",
        value: "As for the math involved, in the upcoming chapters, when we start building models, we'll touch on some mathematical concepts. If you're not familiar with them, or if you've never learned them before, don't stress, I'll break things down visually to make them easier to understand as we go."
      }
    ]
  },
  { id: 2, 
    name: "1. TENSORs", 
    link: "/tensors",
    items: [
      { id: 1, name: "TENSOR ?", link: "tensors" },
      { id: 2, name: "Tensor Creating", link: "tensor-creating" },
      { id: 3, name: "Tensor Attributes", link: "tesor-attributes" },
      { id: 4, name: "Tensor Operations", link: "tensor-operations" },
      { id: 5, name: "Tensor Manipulation", link: "tensor-manipulation" },
      { id: 6, name: "Tensor Indexing", link: "tensor-indexing" },
      { id: 7, name: "Tensor Reproducibility", link: "tensor-reproducibility" },
      { id: 8, name: "Tensor on GPUs", link: "tensor-on-gpus" }
    ],
    description: [
      { id: 1, 
        title: "",
        value: "In this chapter, we're going to talk about TENSOR, which is the most basic and crucial concept in Machine Learning."
      },
      { id: 2, 
        title: "",
        value: "Fist off, you need to know what a TENSOR is. And there is one thing you should always remember - just one thing - 'Computers love DATA'. We use TENSOR to convert everything around us into DATA, whether it's a word, a sound, an image or a video."
      },
      { id: 3, 
        title: "",
        value: "Next up, we'll talk about how to create a tensor and go over its 4 key attributes: shape, dimensions, data type and device. Then we'll move on to tensor operations, like adding, subtracting, dividing, and multiplying. And we'll also cover Matrix Multiplication, which might be a little tricky for beginners but super important."
      },
      { id: 4, 
        title: "",
        value: "After that, you'll learn how to work with tensors. We're going to cover things like aggregating, reshaping, squeezing, unsqueezing, permuting, changing data type, and how to concatenate or stack tensors. We'll also talk about transposing tensors, which is key when you need to fix shape errors during Matrix Multiplication."
      },
      { id: 5, 
        title: "",
        value: "And once you get the hang of tensor indexing, you'll be able to confidently access specific elements - whether it's a single number, a row, rows, a column, columns, a block or specific numbers. We'll also touch on the concept of Reproducibility in Machine Learning, which is super important since models usually start with random data. Finally, we'll wrap up by talking about GPUs, the engines of Machine Learning. We need it."
      }
    ]
   },
  { id: 3, 
    name: "2. A Line Model", 
    link: "/a-line-model",
    items: [
      { id: 1, name: "PyTorch Workflow", link: "pytorch-workflow" },
      { id: 2, name: "i. Prepare Data", link: "prepare-data" },
      { id: 3, name: "ii. Build a Model", link: "build-a-model" },
      { id: 4, name: "iii. Train a Model", link: "train-model" },
      { id: 5, name: "iv. Save a Model", link: "save-model" },
    ],
    description: [
      { id: 1, 
        title: "",
        value: "In this chapter, we will explore the essential steps involved in a typical PyTorch workflow for deep learning, using a straight line model as our guide. Starting with data preparation, we’ll walk through the process of building, training, and evaluating this simple model, making it easier to grasp the key concepts."
      },
      { id: 2, 
        title: "",
        value: "This approach will provide a clear and practical understanding of each stage in the workflow, from data handling to model refinement, all within the context of a linear model that is both accessible and illustrative. By the end of this chapter, you’ll be equipped to apply these principles to more complex models with confidence."
      },
    ]
  },
  { id: 4, 
    name: "3. The Maths Behind (I)", 
    link: "/the-maths-behind-one",
    items: [
      { id: 1, name: "- Linear Regression", link: "linear-regression" },
      { id: 2, name: "- Normal Distribution", link: "normal-distribution" },
      { id: 3, name: "- Loss Function (MSE)", link: "loss-function" },
      { id: 4, name: "- Gradient Descent (GD)", link: "gradient-descent" },
      { id: 5, name: "- Stochastic Gradient Descent (SGD)", link: "stochastic-gradient-descent" },
      { id: 6, name: "- Learning Rate (lr)", link: "learning-rate" }
    ],
    description: [
      { id: 1, 
        title: "",
        value: "We'll start with Linear Regression, the foundation of our straight-line model, which helps us find the best-fitting line to represent the relationship between input and output data. Alongside this, we’ll touch on the Normal Distribution to understand how data points cluster around a mean, which is crucial for statistical analysis."
      },
      { id: 2, 
        title: "",
        value: "Next, we’ll discuss the Loss Function, particularly Mean Squared Error (MSE), which measures how well our model fits the data, and how Loss Curves help visualize this improvement. We'll then explore Gradient Descent, a technique for adjusting our model to minimize the loss, and its faster variant, Stochastic Gradient Descent (SGD)."
      },
      { id: 3, 
        title: "",
        value: "Finally, we'll cover the importance of the Learning Rate, which controls how quickly our model adjusts during training, ensuring that it converges efficiently without overshooting the optimal solution."
      }
    ]
  },
  { id: 5, 
    name: "4. A Classification Model", 
    link: "/a-classification-model",
    items: [
      { id: 1, name: "i. Prepare Data", link: "prepare-data" },
      { id: 2, name: "ii. Build a Model", link: "build-a-model" },
      { id: 3, name: "iii. Train a Model", link: "train-a-model" },
      { id: 4, name: "iv. Improve a Model", link: "improve-a-model" },
      { id: 5, name: "v. Save a Model", link: "save-a-model" },
    ],
    description: [
      { id: 1, 
        title: "",
        value: "In the mathematic part of this chapter, we’ll focus on the classification problems, particularly a binary classification. We’ll start by introducing a new loss function, binary cross-entropy, commonly used in classification tasks."
      },
      { id: 2, 
        title: "",
        value: "Next, we’ll delve into logistic regression, which transforms a set of data into a probability distribution between 0 and 1 - a crucial element in classification models."
      },
      { id: 3, 
        title: "",
        value: "In the last chapter, we were introduced to backpropagation, but we didn’t dive into the details, here we’re gonna explore more about it."
      },
      { id: 4, 
        title: "",
        value: "Finally, we’ll introduce the ReLU function, a non-linear activation function widely used in neural networks."
      },
    ]
  },
  { id: 6, 
    name: "5. The Maths Behind (II)", 
    link: "/the-maths-behind-two",
    items: [
      { id: 1, name: "- Classification Task", link: "classification-task" },
      { id: 2, name: "- Loss Function (BCE)", link: "loss-function-bce" },
      { id: 3, name: "- Sigmoid Function", link: "sigmoid-function" },
      { id: 4, name: "- BackPropagation", link: "backpropagation" },
      { id: 5, name: "- Activation Function (ReLU)", link: "activation-function-relu" }
    ],
    description: [
      { id: 1, 
        title: "",
        value: "In this chapter, we’ll dive into the math and basic machine learing concepts behind the binary classification model we built in the last chapter."
      },
      { id: 2, 
        title: "",
        value: "Let's keep it simple - first, we'll introduce the fundmental concept of classification in machine learning. Then we'll cover a few key functions used in our model, like binary cross-entropy with sigmoid, which together form our loss function. As you might remember, we also introduced the idea of activation functions, and here we use a non-linear one - ReLU. After that, we'll explore the core algrithm - backpropagation, discussing its basic concepts and some implementation details. So, let's get started!",
      },
    ]
  },
  { id: 7, 
    name: "6. A CNN Model", 
    link: "/a-cnn-model",
    items: [
      { id: 1, name: "i. Prepare Data", link: "prepare-data" },
      { id: 2, name: "ii.Build a Model", link: "build-a-model" },
      { id: 3, name: "iii.Train a Model", link: "train-a-model" },
      { id: 4, name: "iv. Improve a Model", link: "improve-a-model" },
      { id: 5, name: "v. Save a Model", link: "save-a-model" },
    ],
    description: [
      { id: 1, 
        title: "",
        value: ""
      },
      { id: 2, 
        title: "",
        value: ""
      },
    ]
  },
  { id: 8, 
    name: "7. The Maths Behind (III)", 
    link: "/the-maths-behind-three",
    items: [
      { id: 1, name: "- Computer Vision", link: "computer-vision" },
      { id: 2, name: "- Image Encoder", link: "image-encoder" },
      { id: 3, name: "- MNIST Dataset", link: "mnist-dataset" },
      { id: 4, name: "- DATALOADER", link: "dataloader" },
      { id: 5, name: "- ARGMAX Function", link: "argmax-function" },
      { id: 6, name: "- CNN", link: "cnn" },
      { id: 7, name: "- MaxPooling Layer", link: "maxPooling-layer" },
      { id: 8, name: "- SoftMax Function", link: "softMax-function" }
    ],
    description: [
      { id: 1, 
        title: "",
        value: ""
      },
      { id: 2, 
        title: "",
        value: ""
      },
    ]
  },
  { id: 9, 
    name: "8. A Vision Transformer Model", 
    link: "/a-vit-model",
    items: [
      { id: 1, name: "i. Prepare Data", link: "prepare-data" },
      { id: 2, name: "ii. Build a Model", link: "build-da-model" },
      { id: 3, name: "iii. Train a Model", link: "train-a-model" },
      { id: 4, name: "iv. Improve a Model", link: "improve-a-model" },
      { id: 5, name: "v. Save a Model", link: "save-a-model" },
    ],
    description: [
      { id: 1, 
        title: "",
        value: ""
      },
      { id: 2, 
        title: "",
        value: ""
      },
    ]
  },
  { id: 10, 
    name: "9. The Maths Behind (IV)", 
    link: "/the-maths-behind-four",
    items: [
      { id: 1, name: "- Vision Transformer", link: "vision-transformer" },
      { id: 2, name: "- ViT Architecture", link: "vit-arhitecture" },
      { id: 3, name: "- Patch Embedding", link: "patch-embedding" },
      { id: 4, name: "-- Class Token Embedding", link: "class-token-embedding" },
      { id: 5, name: "- Positional Embedding", link: "positional-embedding" },
      { id: 6, name: "- Transformer Encoder", link: "transformer-encoder" },
      { id: 7, name: "-- Multi-Head Attention (MHA)", link: "multi-head-self-attention" },
      { id: 8, name: "-- Layer Normalization (LN)", link: "layer-normalization" },
      { id: 9, name: "-- Residual Connection (+)", link: "residual-connection" },
      { id: 10, name: "-- Multi-Layer Perceptron (MLP)", link: "multi-layer-perceptron" },
      { id: 11, name: "-- Dropout Layer", link: "dropout-layer" },
      { id: 12, name: "- Classifier", link: "classifier" },
      { id: 13, name: "- GeLU Function", link: "activation-function" },
    ],
    description: [
      { id: 1, 
        title: "",
        value: ""
      },
      { id: 2, 
        title: "",
        value: ""
      },
    ]
  },
  { id: 11, 
    name: "10. A Language Translation Model", 
    link: "/a-translation-model",
    items: [
      { id: 1, name: "i. Prepare Data", link: "prepare-data" },
      { id: 2, name: "ii. Build a Model", link: "build-a-model" },
      { id: 3, name: "iii. Train and Test a Model", link: "train-a-model" },
      { id: 4, name: "iv. Improve a Model", link: "improve-a-model" },
      { id: 5, name: "v. Save and Load a Model", link: "save-a-model" },
    ],
    description: [
      { id: 1, 
        title: "",
        value: ""
      },
      { id: 2, 
        title: "",
        value: ""
      },
    ]
  },
  { id: 12, 
    name: "11. The Maths Behind (V)", 
    link: "/the-maths-behind-five",
    items: [
      { id: 1, name: "- Word Embedding", link: "word-embedding" },
      { id: 2, name: "- Recurrent Neural Network (RNN)", link: "rnn"},
      { id: 3, name: "- Long Short-Term Memory (LSTM)", link: "lstm"},
      { id: 4, name: "- Transformer", link: "transformer" },
      { id: 5, name: "- Masked Multi-Head Attention", link: "masked-multi-head-attention" },
      { id: 6, name: "- Transformer Encoder", link: "transformer-encoder" },
      { id: 7, name: "- Transformer Decoder", link: "transformer-decoder" },
    ],
    description: [
      { id: 1, 
        title: "",
        value: ""
      },
      { id: 2, 
        title: "",
        value: ""
      },
    ]
  },
  { id: 13, 
    name: "Reference", 
    link: "/reference", 
    items: [], 
    description: [] 
  },
  { id: 14, 
    name: "About the Shape", 
    link: "/about-shape", 
    items: [],
    description: [] 
  },
]