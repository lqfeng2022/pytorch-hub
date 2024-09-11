export default [
  { id: 0, 
    name: "介绍", 
    link: '/introduction', 
    items: [], 
    description: [] 
  },
  { id: 1,
    name: "0. 人工智能 (AI)",
    link: "/artificial-intelligence",
    items: [
      { id: 1, name: "人工智能 (AI)", link: "artificial-intelligence" },
      { id: 2, name: "机器学习", link: "machine-learing" },
      { id: 3, name: "深度学习", link: "deep-learning"},
      { id: 4, name: "神经网络", link: "neural-network" },
      { id: 5, name: "框架和库", link: "libraries" },
      { id: 6, name: "PyTorch", link: "pytorch" },
      { id: 7, name: "前期准备", link: "prerequisites" },
      { id: 8, name: "如何学习这门课衬托", link: "how-to-take-course" },
    ],
    description: [
      { id: 1, 
        title: "",
        value: "在本章中，我们将深入探讨人工智能的基本概念，并介绍一些重要的AI框架和库。首先，我们需要弄清楚什么是人工智能、机器学习、深度学习和神经网络。接下来，我会向大家介绍一些实用的机器学习框架和库。对于初学者来说，PyTorch 是一个不错的选择，我们可以使用它快速构建一个深度学习模型, 因此我们将在接下来的项目中使用 PyTorch。"
      },
      { id: 2, 
        title: "",
        value: "在开始之前，了解一些Python的基础知识很有必要，比如变量、函数、类、数据类型等等。但如果你还不太熟悉Python，不用担心，Python的教学资料很多，你可以很容易学习到, 我相信你会很快掌握的。"
      },
      { id: 3, 
        title: "",
        value: "至于涉及到的数学概念，在接下来的章节中，当我们开始构建模型的时候，会涉及到一些数学内容。如果你不太熟悉或者从未学过，也不用担心，我会通过将这些数学部分放在模型构建章节的后一个章节单独详细解说。"
      },
    ]
  },
  { id: 2, 
    name: "1. TENSORs", 
    link: "/tensors",
    items: [
      { id: 1, name: "张量?", link: "tensors" },
      { id: 2, name: "创建张量", link: "tensor-creating" },
      { id: 3, name: "张量的四大属性", link: "tesor-attributes" },
      { id: 4, name: "张量运算", link: "tensor-operations" },
      { id: 5, name: "张量操作", link: "tensor-manipulation" },
      { id: 6, name: "张量查找", link: "tensor-indexing" },
      { id: 7, name: "张量的可复现性", link: "tensor-reproducibility" },
      { id: 8, name: "GPU上的张量", link: "tensor-on-gpus" }
    ],
    description: [
      { id: 1, 
        title: "",
        value: "在本章中，我们将讨论张量（TENSOR），这是机器学习中的一个基本概念。首先，你需要知道什么是张量, 在计算机中我们用张量来表示各种复杂度的数据, 而数据可以是一个词、一个声音、一张图片或者一个视频等等。"
      },
      { id: 2, 
        title: "",
        value: "接下来，我们将讨论如何创建一个张量，并介绍它的四个关键属性：形状、维度、数据类型和运行环境。然后，我们会说说张量的一些基本运算，比如加法、减法、除法、乘法以及矩阵乘法。"
      },
      { id: 3, 
        title: "",
        value: "之后，你将学习如何操作张量。我们将讨论聚合、重塑、压缩、扩展、排列、更改数据类型以及如何连接或堆叠张量。我们还会讨论张量转置，常见于矩阵乘法。"
      },
      { id: 4, 
        title: "",
        value: "当你学会了张量索引后，你将能够自如地访问张量的特定元素, 无论是一个单独的数字、一行、几行、一列、几列、一部分连续的或者某个特定的数字。我们还会聊聊机器学习中的可重复性概念，这非常重要，因为模型通常以随机数据开始。最后，我们将结束讨论GPU，机器学习的引擎。我们需要它。"
      },
    ]
   },
  { id: 3, 
    name: "2. 一个线性模型", 
    link: "/a-line-model",
    items: [
      { id: 1, name: "PyTorch 工作流程", link: "pytorch-workflow" },
      { id: 2, name: "i. 准备数据", link: "prepare-data" },
      { id: 3, name: "ii. 构建模型", link: "build-a-model" },
      { id: 4, name: "iii. 训练模型", link: "train-model" },
      { id: 5, name: "iv. 保存模型", link: "save-model" },
    ],
    description: [
      { id: 1, 
        title: "",
        value: "在本章中，我们将使用一个典型的 PyTorch 工作流程来构建一个线性模型。从数据准备开始，我们将逐步讲解如何构建一个模型、训练和评估这个模型, 以及如何保存和下载使用这个模型。本章的重点是 PyTorch 工作流程, 至于模型背后的数学原理我们会在下一个章节详细讨论。"
      },
    ]
  },
  { id: 4, 
    name: "3. 模型背后的数学原理(I)", 
    link: "/the-maths-behind-one",
    items: [
      { id: 1, name: "- 线性回归", link: "linear-regression" },
      { id: 2, name: "- 正态分布", link: "normal-distribution" },
      { id: 3, name: "- 损失函数: MSE", link: "loss-function" },
      { id: 4, name: "- 梯度下降(GD)", link: "gradient-descent" },
      { id: 5, name: "- 随机梯度下降(SGD)", link: "stochastic-gradient-descent" },
      { id: 6, name: "- 学习率(lr)", link: "learning-rate" }
    ],
    description: [
      { id: 1, 
        title: "",
        value: "我们首先从线性回归开始，这是我们直线模型的基础，它可以帮助我们找到最佳拟合线来表示输入和输出数据之间的关系。与此同时，我们会提到正态分布，以理解数据点如何围绕均值聚集，这对于统计分析非常关键。"
      },
      { id: 2, 
        title: "",
        value: "接下来，我们将讨论损失函数，特别是均方误差（MSE），它可以衡量我们的模型与目标数据的契合度，另外我们会聊聊损失曲线, 他可以来可视化我们模型的训练过程。然后我们会详细探讨梯度下降法，这是一种典型的模型优化算法, 以及更常用的随机梯度下降法（SGD）。最后，我们会讲讲学习率，它可以控制模型在训练过程的步伐, 快慢，很重要的一个超级参数。"
      },
    ]
  },
  { id: 5, 
    name: "4. 一个二元分类模型", 
    link: "/a-classification-model",
    items: [
      { id: 1, name: "i. 准备数据", link: "prepare-data" },
      { id: 2, name: "ii. 构建模型", link: "build-a-model" },
      { id: 3, name: "iii. 训练模型", link: "train-a-model" },
      { id: 4, name: "iv. 微调模型", link: "improve-a-model" },
      { id: 5, name: "v. 保存模型", link: "save-a-model" },
    ],
    description: [
      { id: 1, 
        title: "",
        value: "这里我们继续使用 PyTorch 工作流程来构建一个二元分类模型。一条直线怎么看都有些无聊, 这一章节, 我们将构建一个多层深度学习模型来区分两组呈圆环状分布的数据集。另外在此工作流程中我们加入了模型微调, 这是我们这一章节探讨的重点。当我们测试我们的第一个模型的时候, 我们发现预测效果很糟糕, 偏差很大, 这里我们通过追加一个非线性方程, 增加更多的训练次数和调整学习率参数这三个方面来优化我们的模型。通过这三次模型调整, 我们成功的将我们的模型预测能力从原本的50%提高到几近于100%, 一次很成功的尝试, 不是吗。"
      },
    ]
  },
  { id: 6, 
    name: "5. 模型背后的数学原理 (II)", 
    link: "/the-maths-behind-two",
    items: [
      { id: 1, name: "- 分类任务", link: "classification-task" },
      { id: 2, name: "- 损失函数 (二元交叉熵 BCE)", link: "loss-function-bce" },
      { id: 3, name: "- Sigmoid函数", link: "sigmoid-function" },
      { id: 4, name: "- 反向传播算法", link: "backpropagation" },
      { id: 5, name: "- 激活方程 (ReLU)", link: "activation-function-relu" }
    ],
    description: [
      { id: 1, 
        title: "",
        value: "在本章中，我们将深入探讨上一章中构建的二元分类模型背后的数学原理和基础机器学习概念。让我们简单点 - 首先，我们将介绍机器学习中分类的基本概念。接着，我们会讲解模型中使用的一些重要函数，比如与Sigmoid函数结合的二元交叉熵，它们共同构成我们的损失函数。你可能还记得，我们之前也介绍过激活函数的概念，在这里我们使用的是一个非线性的激活函数-ReLU。之后，我们将深入探讨深度学习的核心算法-反向传播，讨论其基本概念和一些实现细节。那么，让我们开始吧！"
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
  { id: 12, 
    name: "11. The Maths Behind (V)", 
    link: "/the-maths-behind-five",
    items: [
      { id: 1, name: "- Word Embedding and Word2Vec", link: "word-embedding" },
      { id: 2, name: "- Recurrent Neural Network (RNN)", link: "rnn"},
      { id: 3, name: "- Long Short-Term Memory (LSTM)", link: "lstm"},
      { id: 4, name: "- Transformer", link: "transformer" },
      { id: 5, name: "-- Masked Multi-Head Attention", link: "masked-multi-head-attention" },
      { id: 6, name: "-- Transformer Encoder", link: "transformer-encoder" },
      { id: 7, name: "-- Transformer Decoder", link: "transformer-decoder" },
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