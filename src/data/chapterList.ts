export default [
  { name: "Introduction", link: 'introduction' },
  { id: 0,
    name: "Artificial Intelligence",
    link: "artificial-intelligence",
    items: [
      { id: 1, name: "Artificial Intelligence", link: "artificial-intelligence" },
      { id: 2, name: "Machine Learning", link: "machine-learing" },
      { id: 3, name: "Deep Learning", link: "deep-learning"},
      { id: 4, name: "Neural Network", link: "neural-network" },
      { id: 5, name: "ML Frameworks and Libraries", link: "libraries" },
      { id: 6, name: "PyTorch", link: "pytorch" },
      { id: 7, name: "Prerequisites", link: "prerequisites" },
      { id: 8, name: "How to take this Course", link: "how-to-take-course" },
    ],
    description: [
      { id: 1, 
        title: "",
        value: "In this chapter, we're gonna dive into some fundamental concepts of Artificial Intelligence and explore a few key AI frameworks and libraries."
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
  { id: 1, 
    name: "TENSORs", 
    link: "tensors",
    items: [
      { id: 1, name: "TENSOR", link: "" },
      { id: 2, name: "Create Tensor", link: "" },
      { id: 3, name: "Tensor's Attributes", link: "" },
      { id: 4, name: "Tensor Operation", link: "" },
      { id: 5, name: "Tensor Manipulating", link: "" },
      { id: 6, name: "Tensor Transpose", link: "" },
      { id: 7, name: "Tensor Indexing", link: "" },
      { id: 8, name: "Tensor vs. NumPy Array", link: "" },
      { id: 9, name: "Tensor Reproducibility", link: "" },
      { id: 10, name: "Tensor on GPUs", link: "" }
    ],
    description: [
      { id: 1, 
        title: "",
        value: "In this chapter, we’re gonna talk about the TENSOR, which is the most basic and crucial concept in Machine learning."
      },
      { id: 2, 
        title: "",
        value: "Fist of all, you gotta know what is TENSOR and why tensor is so crucial. There is one thing you should always remember, just one thing - “Computer love data” - , and we use TENSOR to convert everything around our into DATA, whether it’s letters, words, articles, or even voices, images and videos. Next, we’re gonna cover how to create a tensor and go over its 4 key attributes: shape, dimensions, data type and device. Then we’ll move on to tensor operations, adding, subtracting, dividing, and multiplying - just like regular math. We’ll also tackle matrix multiplication, which is a bit tricky for beginners, but super important."
      },
      { id: 3, 
        title: "",
        value: "After that, you’ll learn how to manipulate tensors, we’re gonna talk about aggregating, reshaping, squeezing, unsqueezing, permuting a tensor, changing data types, and even how to concatenate or stack tensors.Knowing how to transpose a tensor is another crucial skill, especially for fixing shape errors during matrix multiplication."
      },
      { id: 4, 
        title: "",
        value: "And once you get the hang of tensor indexing, you’ll be able to confidently access specific elements - whether it’s a single number, row, rows, column, columns, or a block of numbers. We’re gonna also touch on the concept of reproducibility in machine learning, which is super important since models usually starts with random data. Understanding reproducibility and how to ensure it is a big deal. Actually it’s quite simple, you’ll learn about it. Finally, we’ll wrap up by talking about GPUs, the engine of machine learning. We need it."
      }
    ]
   },
  { id: 2, 
    name: "A Straight Line Model", 
    link: "a-straight-line-model",
    items: [
      { id: 1, name: "- PyTorch Workflow", link: "" },
      { id: 2, name: "i. Prepare Data", link: "" },
      { id: 3, name: "ii. Build a Model", link: "" },
      { id: 4, name: "iii. Train and Test a Model", link: "" },
      { id: 5, name: "iv. Improve a Model", link: "" },
      { id: 6, name: "v. Save and Load a Model", link: "" },
    ]
  },
  { id: 3, 
    name: "The Maths Behind (I)", 
    link: "the-maths-behind-one",
    items: [
      { id: 1, name: "- Linear Regression", link: "" },
      { id: 2, name: "- Loss Function (MSA)", link: "" },
      { id: 3, name: "- Gradient Descent (GD)", link: "" },
      { id: 4, name: "- Stochastic Gradient Descent (SGD)", link: "" },
      { id: 5, name: "- Learning Rate (lr)", link: "" }
    ]
  },
  { id: 4, 
    name: "A Binary Classification Model", 
    link: "a-binary-classification-model",
    items: [
      { id: 1, name: "i. Prepare Data", link: "" },
      { id: 2, name: "ii. Build a Model", link: "" },
      { id: 3, name: "iii. Train and Test a Model", link: "" },
      { id: 4, name: "iv. Improve a Model", link: "" },
      { id: 5, name: "v. Save and Load a Model", link: "" },
    ]
  },
  { id: 5, 
    name: "The Maths Behind (II)", 
    link: "the-maths-behind-two",
    items: [
      { id: 1, name: "- Classification Problem", link: "" },
      { id: 2, name: "- Binary Classification Problem", link: "" },
      { id: 3, name: "- Loss Function (BCE)", link: "" },
      { id: 4, name: "- Sigmoid Function", link: "" },
      { id: 5, name: "- BackPropagation", link: "" },
      { id: 6, name: "- Activation Function (ReLU)", link: "" }
    ]
  },
  { id: 6, 
    name: "A CNN Model", 
    link: "a-cnn-model",
    items: [
      { id: 1, name: "i. Prepare Data", link: "" },
      { id: 2, name: "ii.Build a Model", link: "" },
      { id: 3, name: "iii.Train and Test a Model", link: "" },
      { id: 4, name: "iv. Improve a Model", link: "" },
      { id: 5, name: "v. Save and Load a Model", link: "" },
    ]
  },
  { id: 7, 
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
    ]
  },
  { id: 8, 
    name: "A Vision Transformer Model", 
    link: "a-vit-model",
    items: [
      { id: 1, name: "i. Prepare Data", link: "" },
      { id: 2, name: "ii. Build a Model", link: "" },
      { id: 3, name: "iii. Train and Test a Model", link: "" },
      { id: 4, name: "iv. Improve a Model", link: "" },
      { id: 5, name: "v. Save and Load a Model", link: "" },
    ]
  },
  { id: 9, 
    name: "The Maths Behind (IV)", 
    link: "the-maths-behind-four",
    items: [
      { id: 1, name: "- Vision Transformer", link: "" },
      { id: 2, name: "- ViT Architecture", link: "" },
      { id: 3, name: "- Patch Embedding (Epat)", link: "" },
      { id: 4, name: "-- Class Token Embedding", link: "" },
      { id: 5, name: "- Positional Embedding (Epos)", link: "" },
      { id: 6, name: "- Transformer Encoder", link: "" },
      { id: 7, name: "-- Attention Mechanism", link: "" },
      { id: 8, name: "-- Multi-Head Self-Attention (MHA)", link: "" },
      { id: 9, name: "-- Layer Normalization (LN)", link: "" },
      { id: 10, name: "-- Residual Connection (+)", link: "" },
      { id: 11, name: "-- Multi-Layer Perceptron (MLP)", link: "" },
      { id: 12, name: "-- Perceptron", link: "" },
      { id: 13, name: "-- Dropout Layer", link: "" },
      { id: 14, name: "- Classifier", link: "" },
      { id: 15, name: "- Activation Function (GeLU)", link: "" },
    ]
  },
  { id: 10, 
    name: "A Language Translation Model", 
    link: "a-language-translation-model",
    items: [
      { id: 1, name: "i. Prepare Data", link: "" },
      { id: 2, name: "ii. Build a Model", link: "" },
      { id: 3, name: "iii. Train and Test a Model", link: "" },
      { id: 4, name: "iv. Improve a Model", link: "" },
      { id: 5, name: "v. Save and Load a Model", link: "" },
    ]
  },
  { id: 11, 
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
    ]
  },
  { id: 12, name: "Dedication", link: 'dedication' },
  { id: 13, name: "Additional Resources", link: 'additional-resources' },
  { id: 14, name: "Credits", link: 'credits' }
]