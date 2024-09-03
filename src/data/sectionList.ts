export default [
  { id: 0, 
    name: "Introduction", 
    link: 'introduction', 
    items: [], 
    description: [] 
  },
  { id: 1,
    name: "Artificial Intelligence",
    link: "artificial-intelligence",
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
    name: "TENSORs", 
    link: "tensors",
    items: [
      { id: 1, name: "TENSOR ?", link: "" },
      { id: 2, name: "Tensor Creating", link: "" },
      { id: 3, name: "Tensor Attributes", link: "" },
      { id: 4, name: "Tensor Operations", link: "" },
      { id: 5, name: "Tensor Manipulation", link: "" },
      { id: 6, name: "Tensor Indexing", link: "" },
      { id: 7, name: "Tensor Reproducibility", link: "" },
      { id: 8, name: "Tensor on GPUs", link: "" }
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
    name: "A Straight Line Model", 
    link: "a-straight-line-model",
    items: [
      { id: 1, name: "PyTorch Workflow", link: "" },
      { id: 2, name: "i. Prepare Data", link: "" },
      { id: 3, name: "ii. Build a Model", link: "" },
      { id: 4, name: "iii. Train and Test a Model", link: "" },
      { id: 5, name: "iv. Save and Load a Model", link: "" },
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
    name: "The Maths Behind (I)", 
    link: "the-maths-behind-one",
    items: [
      { id: 1, name: "- Linear Regression", link: "" },
      { id: 2, name: "- Normal Distribution", link: "" },
      { id: 3, name: "- Loss Function (MSE)", link: "" },
      { id: 4, name: "- Gradient Descent (GD)", link: "" },
      { id: 5, name: "- Stochastic Gradient Descent (SGD)", link: "" },
      { id: 6, name: "- Learning Rate (lr)", link: "" }
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
    name: "A Binary Classification Model", 
    link: "a-binary-classification-model",
    items: [
      { id: 1, name: "i. Prepare Data", link: "" },
      { id: 2, name: "ii. Build a Model", link: "" },
      { id: 3, name: "iii. Train and Test a Model", link: "" },
      { id: 4, name: "iv. Improve a Model", link: "" },
      { id: 5, name: "v. Save and Load a Model", link: "" },
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
  { id: 6, 
    name: "The Maths Behind (II)", 
    link: "the-maths-behind-two",
    items: [
      { id: 1, name: "- Classification Problem", link: "" },
      { id: 2, name: "- Binary Classification Problem", link: "" },
      { id: 3, name: "- Loss Function (BCE)", link: "" },
      { id: 4, name: "- Sigmoid Function", link: "" },
      { id: 5, name: "- BackPropagation", link: "" },
      { id: 6, name: "- Activation Function (ReLU)", link: "" }
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
  { id: 7, 
    name: "A CNN Model", 
    link: "a-cnn-model",
    items: [
      { id: 1, name: "i. Prepare Data", link: "" },
      { id: 2, name: "ii.Build a Model", link: "" },
      { id: 3, name: "iii.Train and Test a Model", link: "" },
      { id: 4, name: "iv. Improve a Model", link: "" },
      { id: 5, name: "v. Save and Load a Model", link: "" },
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
    name: "The Maths Behind (III)", 
    link: "the-maths-behind-three",
    items: [
      { id: 1, name: "- Computer Vision", link: "" },
      { id: 2, name: "- Image Encoder", link: "" },
      { id: 3, name: "- MNIST Dataset", link: "" },
      { id: 4, name: "- DATALOADER", link: "" },
      { id: 5, name: "- ARGMAX Function", link: "" },
      { id: 6, name: "- Convolutional Neural Network (CNN)", link: "" },
      { id: 7, name: "- MaxPooling Layer", link: "" },
      { id: 8, name: "- SoftMax Function", link: "" }
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
    name: "A Vision Transformer Model", 
    link: "a-vit-model",
    items: [
      { id: 1, name: "i. Prepare Data", link: "" },
      { id: 2, name: "ii. Build a Model", link: "" },
      { id: 3, name: "iii. Train and Test a Model", link: "" },
      { id: 4, name: "iv. Improve a Model", link: "" },
      { id: 5, name: "v. Save and Load a Model", link: "" },
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
    name: "The Maths Behind (IV)", 
    link: "the-maths-behind-four",
    items: [
      { id: 1, name: "- Vision Transformer", link: "" },
      { id: 2, name: "- ViT Architecture", link: "" },
      { id: 3, name: "- Patch Embedding (Epat)", link: "" },
      // { id: 4, name: "-- Class Token Embedding", link: "" },
      { id: 5, name: "- Positional Embedding (Epos)", link: "" },
      { id: 6, name: "- Transformer Encoder", link: "" },
      // { id: 7, name: "-- Attention Mechanism", link: "" },
      // { id: 8, name: "-- Multi-Head Self-Attention (MHA)", link: "" },
      // { id: 9, name: "-- Layer Normalization (LN)", link: "" },
      // { id: 10, name: "-- Residual Connection (+)", link: "" },
      // { id: 11, name: "-- Multi-Layer Perceptron (MLP)", link: "" },
      // { id: 12, name: "-- Perceptron", link: "" },
      // { id: 13, name: "-- Dropout Layer", link: "" },
      { id: 14, name: "- Classifier", link: "" },
      { id: 15, name: "- Activation Function (GeLU)", link: "" },
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
    name: "A Language Translation Model", 
    link: "a-language-translation-model",
    items: [
      { id: 1, name: "i. Prepare Data", link: "" },
      { id: 2, name: "ii. Build a Model", link: "" },
      { id: 3, name: "iii. Train and Test a Model", link: "" },
      { id: 4, name: "iv. Improve a Model", link: "" },
      { id: 5, name: "v. Save and Load a Model", link: "" },
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
    name: "The Maths Behind (V)", 
    link: "the-maths-behind-five",
    items: [
      { id: 1, name: "- Word Embedding", link: "" },
      { id: 2, name: "- Recurrent Neural Network (RNN)", link: ""},
      { id: 3, name: "- Long Short-Term Memory (LSTM)", link: ""},
      { id: 4, name: "- Transformer", link: "" },
      { id: 5, name: "- Masked Multi-Head Attention", link: "" },
      { id: 6, name: "- Transformer Encoder", link: "" },
      { id: 7, name: "- Transformer Decoder", link: "" },
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
    name: "Dedication", 
    link: 'dedication', 
    items: [], 
    description: [] 
  },
  { id: 14, 
    name: "Additional Resources", 
    link: 'additional-resources', 
    items: [],
    description: [] 
  },
  { id: 15, 
    name: "Credits", 
    link: 'credits',
    items: [],
    description: [] 
  }
]