export default [
  { id: 0, 
    name: "Classification Problem",
    sections: [
      { id: 0,
        name: "1. Classification Probelem", 
        value: "A classification problem involves categorizing a object into one of n distinct classes.",
        image: "",
        content: [
          { id: 1, 
            title: "",
            value: "分类问题是一种机器学习问题, 其目标是将输入分配到几个预定义的类别或类之一。基本上, 我们分析一个对象或数据点, 并根据其特征将其归入 n 个可能类别中的一个。"
          },
          { id: 2, 
            title: "",
            value: "例如, 在二分类问题中, 我们只有两个可能的类别, 比如电子邮件过滤器中的‘垃圾邮件’或‘非垃圾邮件’。对于多分类问题, 可能有多个选项, 比如决定一张图片是狗、猫还是鸟。分类算法的主要目标是从训练数据中学习, 并通过从已经学习的示例中泛化, 对新的、未见过的数据做出准确的预测。"
          },
        ]
      },
      { id: 1,
        name: "1.1 Classification Probelm", 
        value: "",
        image: "src/assets/chapter_five/classific.jpeg",
        content: [
          { id: 1, 
            title: "二元分类",
            value: "二元分类是最简单的分类形式, 其中输入被分配到两个不同的类别之一。它基本上是一个只有两个可能结果的决策过程。一个常见的例子是垃圾邮件检测, 其中电子邮件被分类为“垃圾邮件”或“非垃圾邮件”。"
          },
          { id: 2, 
            title: "多元分类",
            value: "多元分类涉及将输入分配到两个以上的类别之一。与只有两个选项的二元分类不同, 多分类处理多个类别的问题。一个很好的例子是图像识别, 其中图像被分类为“猫”、“狗”或“鸟”等其他可能性。"
          },
          { id: 3, 
            title: "多标签分类",
            value: "多标签分类允许一个输入被分配多个标签。与每个输入仅与一个类别相关的二元分类和多分类不同, 在多标签分类中, 一个输入可以同时属于多个类别。一个很好的例子是电影类型分类, 其中一部电影可以同时被标记为“动作片”、“喜剧片”和“惊悚片”等。"
          },
        ]
      },
      { id: 2,
        name: "1.2 二元分类", 
        value: "Binary classification is a fundamental task that assigns inputs to one of two distinct categories.",
        image: "",
        content: [
          { id: 1, 
            title: "简单性",
            value: "二元分类是最简单的分类形式, 只涉及两个类别。与更复杂的任务相比, 这种简单性使其更容易理解和实现。"
          },
          { id: 2, 
            title: "构建模块",
            value: "许多复杂的机器学习任务, 例如多分类和多标签分类, 可以看作是二分类问题的扩展或组合。例如, 多分类问题通常可以拆解为多个二分类任务。"
          },
          { id: 3, 
            title: "广泛应用",
            value: "二分类在现实世界中有广泛的应用, 包括垃圾邮件检测、疾病诊断和情感分析等。这些任务在各个行业中都很常见, 因此二分类是许多机器学习系统的关键组成部分。"
          },
          { id: 4, 
            title: "算法基础",
            value: "许多机器学习算法最初都是在二分类任务上开发和测试的, 然后再扩展到处理更复杂的场景。理解二分类对于掌握这些算法的核心概念至关重要。"
          },
        ]
      },
      { id: 3,
        name: ":: 线性关系", 
        value: "",
        image: "src/assets/chapter_five/classific_linear.jpeg",
        content: [
          { id: 1, 
            title: "",
            value: "线性关系意味着在二维空间中, 两个类别可以用一条直线分开；在三维空间中, 则用一个平面分开；在更高维度中, 用超平面分开。"
          },
          { id: 2, 
            title: "",
            value: "这里我们专注于最简单的情况——一条直线。线性模型的设计目标是找到一条能够最好地分开两个类别的直线。"
          },
          { id: 3, 
            title: "",
            value: "线性模型简单易懂, 通常训练速度较快且占用的资源较少。然而, 线性模型也有局限性, 例如无法处理非线性可分的数据。如果两个类别之间的决策边界是曲线或复杂形状, 线性模型可能无法表现良好。",
          }
        ]
      },
      { id: 4,
        name: ":: 非线性关系", 
        value: "",
        image: "src/assets/chapter_five/classific_nonlinear.jpeg",
        content: [
          { id: 1, 
            title: "",
            value: "非线性关系意味着类别无法通过一条直线、一个平面或超平面分开。相反, 决策边界可能呈现为曲线或更复杂的形状。"
          },
          { id: 2, 
            title: "",
            value: "在非线性关系中, 决策边界可以是一条曲线、多条曲线, 甚至更复杂的形状。"
          },
          { id: 3, 
            title: "",
            value: "尽管非线性模型提供了更大的灵活性, 能够捕捉复杂的模式和关系, 使其适用于更广泛的问题, 但它们通常更复杂, 难以解释。非线性模型还需要更多的计算资源, 训练时间也可能更长。"
          },
        ]
      },
    ],
  },
  { id: 1, 
    name: "Loss Function - Binary Cross Entropy",
    sections: [
      { id: 0,
        name: "2. 二元交叉熵", 
        value: "Binary cross-entropy is a popular loss function to measure the performance of binary classification problems.",
        image: "",
        content: [
          { id: 1, 
            title: "",
            value: "二元交叉熵是机器学习中用于二分类任务的常用损失函数。它通过测量预测概率与实际二进制标签 (0 或 1) 之间的差异, 量化模型预测与真实标签的匹配程度。"
          },
          { id: 2, 
            title: "",
            value: "在二分类中, 目标是将每个输入分配到两个可能类别中的一个。二元交叉熵通过惩罚偏离真实标签的预测来评估模型的准确性。低交叉熵损失表明预测概率接近实际类别, 反映了良好的模型性能；而高损失则表明偏差较大, 模型表现较差。"
          },
        ],
      },
      { id: 1,
        name: ":: Binary Cross Entropy Formula", 
        value: "",
        image: "src/assets/chapter_five/bceFormula.jpeg",
        content: [
          { id: 1, 
            title: "",
            value: "交叉熵听起来复杂, 但在二分类中其实很简单。当处理两个类别时, 通常将真实标签设为 1。此时, 二元交叉熵公式简化为 BCE = -log(x), 其中 x 是预测正确的概率, 范围在 0 到 1 之间。"
          },
          { id: 2, 
            title: "",
            value: "如果查看显示预测概率 (x) 与训练损失之间关系的图表, 你将看到这一概念的实际应用。"
          },
        ],
      },
      { id: 2,
        name: ":: Cross Entropy", 
        value: "Cross-entropy measures the difference between two probability distributions.",
        image: "",
        content: [
          { id: 1, 
            title: "",
            value: "交叉熵衡量两个概率分布之间的差异：真实分布 P 和估计分布 Q。无论类别数量如何, 它评估一个概率分布如何近似另一个概率分布。“交叉熵”中的“交叉”一词突出强调了比较或结合这两个不同分布的概念。"
          },
          { id: 2, 
            title: "",
            value: "在二分类问题中, 只有两个可能的类别, 二元交叉熵是交叉熵的特定形式, 专门用于处理每个类别对应的两个分布。"
          },
        ],
      },
      { id: 3,
        name: ":: Entropy", 
        value: "Entropy is a scientific concept often associated with disorder, randomness, or uncertainty.",
        image: "",
        content: [
          { id: 1, 
            title: "",
            value: "熵的概念最早在热力学中引入, 用于衡量系统中的无序或不可预测性。"
          },
          { id: 2, 
            title: "",
            value: "在信息论中, 熵用于量化数据集中的不确定性或随机性。它衡量在数据划分或分类时获得了多少信息, 因此在数据科学和机器学习中是一个关键概念。"
          },
        ],
      },
    ],
  },
  { id: 2, 
    name: "Sigmoid Function",
    sections: [
      { id: 0,
        name: "3. Sigmoid 函数", 
        value: "The sigmoid function takes any real-valued input and maps it to a value between 0 and 1.",
        image: "",
        content: [
          { id: 1, 
            title: "",
            value: "Sigmoid 函数的作用是将任意输入转换为类似概率的输出, 这使其非常适合分类任务。"
          },
          { id: 1, 
            title: "",
            value: "在我们的二元分类模型中, 我们将 Sigmoid 函数与二元交叉熵结合使用。首先, 使用 Sigmoid 函数将输入值转换为 0 到 1 之间的范围。然后, 将这些值传递给二元交叉熵函数以计算损失。之所以这样做, 是因为二元交叉熵是针对类似概率的值设计的, 而这些值的范围在 0 到 1 之间。"
          },
        ]
      },
      { id: 1,
        name: ":: Sigmoid 函数公式", 
        value: "",
        image: "src/assets/chapter_five/sigmoidFormula.jpeg",
        content: [
          { id: 1, 
            title: "S 形曲线",
            value: "其图像呈 S 形：在开始时接近 0, 中间陡然上升, 接近 1 时趋于平缓。"
          },
          { id: 2, 
            title: "输出范围: 0 ~ 1",
            value: "Sigmoid 函数的输出值始终在 0 到 1 之间, 这使其非常适合表示概率。"
          },
          { id: 3, 
            title: "特殊点: (0, 0.5)",
            value: "函数在 x=0 时对称, 输出值为 0.5。"
          },
          { id: 4, 
            title: "应用",
            value: "Sigmoid 函数在基于事件发生概率的决策场景中至关重要, 例如判断电子邮件是否为垃圾邮件。"
          },
        ]
      },
      { id: 2,
        name: ":: Sigmoid 函数的特性", 
        value: "",
        image: "src/assets/chapter_five/sigmoidFeature.jpeg",
        content: [
          { id: 1, 
            title: "概率输出",
            value: "Sigmoid 函数将广泛的输入值映射到 0 到 1 的范围内, 这使我们能够将不同的输入转换为概率分布。"
          },
          { id: 2, 
            title: "适合二元分类",
            value: "Sigmoid 函数在二元分类任务中特别有用, 因为它将输入值压缩到 [0, 1] 的范围内, 有效处理大输入值。"
          },
          { id: 3, 
            title: "梯度消失问题",
            value: "尽管 Sigmoid 函数的梯度计算简单, 但对于极端输入值, 梯度可能非常小, 从而导致训练过程中权重变化极小, 这就是所谓的梯度消失问题。"
          },
          { id: 4,
            title: "昂贵的计算成本",
            value: "Sigmoid 函数涉及指数计算, 因此相较于其他激活函数, 它的计算成本较高, 效率较低。",
          }
        ]
      },
    ],
  },
  { id: 3, 
    name: "ReLU Function",
    sections: [
      { id: 0,
        name: "4. ReLU 函数", 
        value: "ReLU (Rectified Linear Unit) is a non-linear activation function commonly used in deep learning.",
        image: "",
        content: [
          { id: 1, 
            title: "",
            value: "ReLU 函数在深度学习中被广泛使用, 特别是在卷积神经网络 (CNN) 和其他各种神经网络架构中。"
          },
        ]
      },
      { id: 1,
        name: ":: ReLU 函数方程", 
        value: "",
        image: "src/assets/chapter_five/reluFormula.jpeg",
        content: [
          { id: 1, 
            title: "",
            value: "ReLU 函数有一个简单的公式：它输出 0 和输入值 x 之间的最大值。因此, ReLU 的输出总是非负的, 范围从 0 到正无穷。对于正输入, 函数是线性的；对于负输入, 输出为 0。"
          },
          { id: 2, 
            title: "",
            value: "总体来说, ReLU 在许多深度学习模型中受到青睐, 因为它简单、高效, 并且能够加快和提升深度神经网络的训练效果。"
          },
        ]
      },
      { id: 2,
        name: ":: ReLU 函数特性", 
        value: "",
        image: "src/assets/chapter_five/reluFeature.jpeg",
        content: [
          { id: 1, 
            title: "非线性",
            value: "虽然 ReLU 函数简单且分段线性, 但它为网络引入了非线性, 这对学习复杂模式至关重要。"
          },
          { id: 2, 
            title: "计算效率高",
            value: "ReLU 仅激活输入值为正的神经元, 导致稀疏激活。这种稀疏性可以提高计算效率。"
          },
          { id: 3, 
            title: "无梯度消失问题",
            value: "与 sigmoid 或 tanh 函数不同, ReLU 避免了正输入的梯度消失问题, 从而加速训练并在深度网络中提高性能。"
          },
          { id: 4, 
            title: "ReLU 失效问题",
            value: "然而, 如果输入值大多为负数, 神经元可能会变得不活跃, 输出为 0, 实际上停止学习。这一问题被称为“ReLU失效”问题。"
          },
        ]
      },
    ],
  },
  { id: 4,
    name: "BackPropagation",
    sections: [
      { id: 0,
        name: "5. BackPropagation",
        value: "Backpropagation is a key algorithm for training neural networks, which involves the 'backward propagation of errors' to adjust weights and minimize loss.",
        image: "",
        content: [
          { id: 0, 
            title: "",
            value: "反向传播是神经网络训练的一种过程, 通过将预测的误差率 (前向传播的结果) 反向传递回网络, 来调整参数如权重, 从而提高模型的准确性。"
          },
          { id: 1, 
            title: "前向传播",
            value: "在前向传播过程中, 网络生成预测值, 并计算这些预测值与实际目标值之间的差异作为误差。"
          },
          { id: 2, 
            title: "反向传播",
            value: "然后, 误差被逐层反向传播通过网络。在这个过程中, 调整神经元之间连接的权重以最小化误差。"
          },
          { id: 3, 
            title: "循环：前向传播 -> 反向传播",
            value: "通过基于误差持续更新权重, 网络迭代地学习和提高其准确性, 逐渐减少预测值与实际值之间的差异。"
          },
        ]
      },
      { id: 1,
        name: ":: 前向传播:",
        value: "",
        image: "src/assets/chapter_five/backpropagat_implem.jpeg",
        content: [
          { id: 0, 
            title: "",
            value: "为了展示反向传播的概念, 我绘制了一个简单的神经网络, 包含两个隐藏层。这个图示展示了网络的结构, 并说明了前向传播的流程, 其中输入数据通过网络传递以生成预测输出。"
          },
        ]
      },
      { id: 2,
        name: ":: 反向传播:",
        value: "",
        image: "src/assets/chapter_five/backpropagat_calcul.jpeg",
        content: [
          { id: 0, 
            title: "",
            value: "反向传播的主要目标是更新权重和偏置, 以最小化预测输出和期望输出之间的误差或损失。这个过程包括:"
          },
          { id: 1, 
            title: "计算损失",
            value: "损失通常使用均方误差 (MSE) 等函数来衡量。"
          },
          { id: 2, 
            title: "计算梯度",
            value: "反向传播的关键步骤是计算损失函数相对于权重和偏置的梯度。使用链式法则, 我们计算每个权重和偏置对总体误差的贡献, 这涉及计算偏导数。"
          },
          { id: 3, 
            title: "更新权重和偏置",
            value: "在计算梯度之后, 权重和偏置会被更新以减少误差。这通常通过梯度下降法完成, 如前面的模型中所述。"
          },
          { id: 4, 
            title: "重复这一过程",
            value: "前向传播和反向传播步骤会被反复执行, 直到模型收敛, 即损失函数达到最小值, 并且网络能够准确预测给定输入的输出。"
          },
        ]
      },
    ]
  },
]