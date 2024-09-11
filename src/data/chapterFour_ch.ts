export default [
  { id: 0, 
    name: "Prepare DATA",
    sections: [
      { id: 0, 
        name: "1. Prepare DATA", 
        value: "Too much explaining is not a good thing. We’ll explore what data preparation is and how it works in this real project.",
        image: "",
        content: [
        ]
      },
      { id: 1, 
        name: ":: Create DATA", 
        value: "",
        image: "src/assets/chapter_four/prepare_create.jpeg",
        content: [
          { id: 1,
            title: "",
            value: "与我们的第一个项目不同, 在这个项目中, 我们使用Scikit Learn数据集来创建一个圆形分布的数据集。您很快就会看到分布点的可视化。"
          },
        ]
      },
      { id: 2, 
        name: ":: Split DATA", 
        value: "",
        image: "src/assets/chapter_four/prepare_split.jpeg",
        content: [
          { id: 1,
            title: "",
            value: "与之前的模型一样, 我们按常规比例将数据分为训练集和测试集。在这里, 我们将40个数据点 (占总数的80%) 分配给训练集, 而剩下的10个数据点 (占20%) 保留给测试集。"
          },
        ]
      },
      { id: 3, 
        name: ":: Visualize DATA", 
        value: "",
        image: "src/assets/chapter_four/prepare_visual.png",
        content: [
          { id: 1,
            title: "",
            value: "在这里, 我们绘制了数据集, 显示出1000个数据点明显分布在两个不同的簇中, 每个簇呈圆形分布, 一个簇嵌套在另一个簇内。在接下来的步骤中, 我们将构建一个模型来对这些圆形簇进行分类。"
          },
        ]
      },
    ]
  },
  { id: 1, 
    name: "Build a Model",
    sections: [
      { id: 0, 
        name: "2. Build a Model", 
        value: "Building a model in deep learning involves designing, implementing a neural network to learn patterns from data for a specific task.",
        image: "",
        content: [
          { id: 1,
            title: "",
            value: "这一步涉及选择最适合当前问题的模型架构。一旦架构定义完成, 模型会使用像 PyTorch 这样的框架进行实现。这包括设置模型的层次结构、指定激活函数以及初始化权重。随后, 模型将准备好在数据上进行训练, 通过调整参数来最小化误差并提高预测准确性。"
          },
        ]
      },
      { id: 1, 
        name: ":: Construct model Architecture", 
        value: "",
        image: "src/assets/chapter_four/build_architecture.jpeg",
        content: [
          { id: 1,
            title: "",
            value: "为了更好地理解这个模型, 我对其架构进行了可视化。图中清晰展示了结构:一个简单的深度学习模型, 包含三个主要部分——输入层、单一隐藏层和输出层。输入层接收两个特征, 隐藏层由五个神经元组成, 输出层则生成一个特征。就是这么简单！"
          }
        ]
      },
      { id: 2, 
        name: ":: Build a Model with PyTorch", 
        value: "",
        image: "src/assets/chapter_four/build_model.jpeg",
        content: [
          { id: 1,
            title: "",
            value: "首先, 我们通过继承PyTorch的基类创建一个神经网络类。我们定义了两层:layer_1 接收2个输入特征并输出5个特征, layer_2 则将这5个特征减少为1个输出。这两层都使用线性函数。"
          },
          { id: 2,
            title: "",
            value: "接着, 我们重写了 forward() 方法, 指定输入数据 X 如何在网络中传递。输入首先经过 layer_1, 然后通过 layer_2, 最终输出结果。"
          },
          { id: 3,
            title: "",
            value: "最后, 我们实例化了 CircleModelV0 类, 并将模型移动到指定的设备 (CPU或GPU) 上执行。这确保了模型在指定的硬件上运行。"
          },
        ]
      },
    ]
  },
  { id: 2, 
    name: "Train a Model",
    sections: [
      { id: 0, 
        name: "3. Train a Model", 
        value: "Unlike our previous model, we use a new loss function - Binary Cross Entropy - and run training and test data on GPUs if available.",
        image: "",
        content: [
        ]
      },
      { id: 1, 
        name: ":: Build a training loop", 
        value: "",
        image: "src/assets/chapter_four/train_model.jpeg",
        content: [
          { id: 0,
            title: "",
            value: "这个二分类模型与之前的模型有两个主要区别:"
          },
          { id: 1,
            title: "引入新的损失函数",
            value: "我们使用了二元交叉熵 (Binary Cross Entropy) 作为新的损失函数, 这是衡量二分类任务表现的常用方法。相比之前的损失函数, 二元交叉熵更适合处理二分类问题。"
          },
          { id: 2,
            title: "使用GPU进行训练和测试",
            value: "在这个项目中, 如果有GPU可用, 我们将在训练和测试数据中使用GPU进行计算, 以加速模型的训练过程, 提高性能。"
          },
        ]
      },
      { id: 2, 
        name: ":: Test the trained Model", 
        value: "",
        image: "src/assets/chapter_four/train_visual.jpeg",
        content: [
          { id: 1,
            title: "",
            value: "在训练后, 我们在训练数据和测试数据上测试了我们的模型, 得到了两组预测结果。当我们绘制这些预测结果时, 可以明显看到模型的表现很差。图中显示了一条带有轻微角度的直线将数据点分开, 无论这些点是来自训练集还是测试集。"
          },
          { id: 2,
            title: "",
            value: "预测效果真是糟糕透了！这种差劲的表现可能是因为我们在模型中仅使用了线性函数。由于数据点呈圆形分布, 非线性函数可能会更有效, 因为直线无法正确地分开圆形数据对, 对吧？"
          }
        ]
      },
      { id: 3, 
        name: ":: Check loss curves", 
        value: "",
        image: "src/assets/chapter_four/train_loss.jpeg",
        content: [
          { id: 1,
            title: "",
            value: "尽管模型的表现很差, 我们仍然应该检查损失曲线, 作为常规训练过程的一部分。"
          },
          { id: 2,
            title: "",
            value: "从损失曲线中, 我们看到训练损失和测试损失都随着训练轮数的增加而平稳且缓慢地减少。我们可能需要考虑增加训练轮数。然而, 损失在100轮中仅下降了约0.01, 从大约0.7降到0.69。这种缓慢的进展表明, 即使使用1,000轮训练, 模型的表现也可能不会有太大改善。不过, 尝试一下也是值得的, 对吧？"
          },
        ]
      },
    ]
  },
  { id: 3,
    name: "Improve a Model",
    sections: [
      { id: 0, 
        name: "4. Improve a Model", 
        value: "Improving a model involves making adjustments to enhance its performance on a given task.",
        image: "",
        content: [
          { id: 1,
            title: "",
            value: "这可以通过多种方式实现:增加更多层以提升模型的深度, 添加更多隐藏单元以增强模型的容量, 增加训练轮次以获得更多的学习迭代, 调整学习率以改善收敛, 修改损失函数、激活函数或优化器以更好地适应问题, 以及使用预训练模型来利用现有知识。"
          },
        ]
      },
      { id: 1, 
        name: ":: How to Improve a Model", 
        value: "",
        image: "src/assets/chapter_four/improve_model.jpeg",
        content: [
          { id: 0,
            title: "",
            value: "当我们想要改进模型时, 可以考虑两个主要方面:模型本身和训练过程。"
          },
          { id: 1,
            title: "重新构建模型",
            value: "输入层和输出层是固定的, 所以我们可以专注于隐藏层。我们可以增加更多层来使模型更深, 或者增加更多神经元以提升模型的能力。我们还可以考虑更改激活函数 (这一点我们将在数学章节中讨论) 。使用预训练模型也是一个选择, 这在即将到来的项目中将非常有用。"
          },
          { id: 2,
            title: "重新构建训练过程",
            value: "我们也可以使用不同的参数和函数来训练模型。例如, 我们可以调整学习率, 设置更多的训练轮次, 或尝试不同的损失函数或优化器。在这个模型中, 我们将只讨论调整学习率和增加训练轮次。"
          },
        ]
      },
    ]
  },
  { id: 4, 
    name: "The 1st Model (+ReLU)",
    sections: [
      { id: 0, 
        name: "4.1 The First Model (+ReLU)", 
        value: "",
        image: "src/assets/chapter_four/improve_one.jpeg",
        content: [
          { id: 1,
            title: "",
            value: "由于我们到目前为止使用的是线性函数来构建模型, 因此我们还没有探索非线性函数。现在, 我们将引入第一个非线性公式作为该模型的激活函数, 看看它的表现如何。"
          },
        ]
      },
      { id: 1, 
        name: ":: Architecture of Model", 
        value: "",
        image: "src/assets/chapter_four/improve_one_architec.jpeg",
        content: [
          { id: 1,
            title: "",
            value: "在我们重新构建模型之前, 让我们可视化其架构, 并引入一个新的非线性激活函数——ReLU (修正线性单元) 。ReLU 是一个强大的非线性函数, 它会将流向神经元的所有负值消除。"
          }
        ]
      },
      { id: 2, 
        name: ":: Build and Train model", 
        value: "",
        image: "src/assets/chapter_four/improve_one_build.jpeg",
        content: [
          { id: 1,
            title: "",
            value: "首先, 我们添加了一个新的模型参数‘relu’, 即 ReLU 函数。在计算过程中, 我们将其放置在 layer_1 和 layer_2 之间, 以过滤掉来自 layer_1 的负值。"
          }
        ]
      },
      { id: 3, 
        name: ":: Evaluate Model", 
        value: "",
        image: "src/assets/chapter_four/improve_one_test.png",
        content: [
          { id: 1,
            title: "",
            value: "在训练和测试这个新模型后, 我们将预测结果与训练数据和测试数据进行对比绘图。从这个图像中, 我们可以清楚地看到, 即使添加了非线性函数, 我们的预测效果仍然很差。这是为什么呢？我们是否需要更多的训练周期？让我们继续探索。"
          }
        ]
      },
      { id: 4, 
        name: ":: Check the Loss Curves", 
        value: "",
        image: "src/assets/chapter_four/improve_one_loss.png",
        content: [
          { id: 1,
            title: "",
            value: "尽管模型效果仍然很差, 我们还是应该检查损失曲线——也许我们可以从另一个角度发现一些问题。"
          },
          { id: 2,
            title: "",
            value: "与我们原始模型相比, 这些曲线变化持续, 几乎与训练周期的增加速度相同。损失仍在缓慢下降, 很像原始模型的情况。"
          },
        ]
      },
    ]
  },
  { id: 5, 
    name: "The 2nd Model (+epochs)",
    sections: [
      { id: 0, 
        name: "4.2 The Second Model (+epochs)", 
        value: "",
        image: "src/assets/chapter_four/improve_two.jpeg",
        content: [
          { id: 1,
            title: "",
            value: "在这个模型中, 我们保持了第一个模型的架构。然而, 在训练过程中, 我们将训练周期的数量从100增加到1,000, 以进行更多的学习迭代。现在, 让我们构建这个新模型——model_2——训练它, 然后评估其性能。"
          },
        ]
      },
      { id: 1, 
        name: ":: Build and Train Model", 
        value: "",
        image: "src/assets/chapter_four/improve_two_build.jpeg",
        content: [
          { id: 1,
            title: "",
            value: "在这个实验中, 我们保持了与model_1相同的模型架构。由于我们只是重建相同的模型, 所以这里不包括代码——只需参考之前的代码, 当然, 模型名称已更新。"
          },
          { id: 2,
            title: "",
            value: "这张图显示了训练循环设置, 其中唯一的变化是将训练周期从100增加到1,000。"
          }
        ]
      },
      { id: 2, 
        name: ":: Evaluate Model", 
        value: "",
        image: "src/assets/chapter_four/improve_two_test.png",
        content: [
          { id: 1,
            title: "",
            value: "查看这个图, 我们可以看到预测结果非常不错——比上一个模型要好得多。几乎80%的点都被白色曲线分隔开, 这是一种很好的迹象。这意味着我们正在朝着正确的方向前进。"
          }
        ]
      },
      { id: 3, 
        name: ":: Check the Loss Curves", 
        value: "",
        image: "src/assets/chapter_four/improve_two_loss.png",
        content: [
          { id: 1,
            title: "",
            value: "从这张图中, 我们可以看到损失曲线很平滑, 损失值随着训练轮次的增加而稳步下降。在训练进行到一半时, 曲线开始变得更加陡峭, 直到训练结束。因此, 如果我们增加更多的训练轮次, 这条曲线很可能会以类似的速度继续下降。"
          }
        ]
      },
    ]
  },
  { id: 6, 
    name: "The 3rd Model (->lr)",
    sections: [
      { id: 0, 
        name: "4.3 The Third Model (->lr)", 
        value: "",
        image: "src/assets/chapter_four/improve_three.jpeg",
        content: [
          { id: 1,
            title: "",
            value: "在模型_1 和模型_2 的基础上, 我们现在将重点关注学习率——这是深度学习中一个至关重要的超参数。我们的前两个模型在损失变化上较慢, 那么如果我们增加学习步长会怎样呢？通过增大学习步长, 我们可能在相同的训练轮次和相同的架构下获得更好的模型。让我们来进行这个实验。"
          },
        ]
      },
      { id: 1, 
        name: ":: Build and Train Model", 
        value: "",
        image: "src/assets/chapter_four/improve_three_build.jpeg",
        content: [
          { id: 1,
            title: "",
            value: "接下来, 我们将构建一个新模型——模型_3。在训练循环中, 我们调整了学习率, 以实现更快的学习步伐。让我们训练并评估这个新模型。"
          }
        ]
      },
      { id: 2, 
        name: ":: Evaluate Model", 
        value: "",
        image: "src/assets/chapter_four/improve_three_test.png",
        content: [
          { id: 1,
            title: "",
            value: "哇, 从图中我们可以清楚地看到, 模型成功地将两个点簇分开, 即使它们是圆形的。预测准确率接近 100%——这是一个完美的模型！"
          }
        ]
      },
      { id: 3, 
        name: ":: Check the Loss Curves", 
        value: "",
        image: "src/assets/chapter_four/improve_three_loss.png",
        content: [
          { id: 1,
            title: "",
            value: "从这些损失曲线中, 我们可以清楚地看到, 尽管训练轮数相同, 但这个模型的损失比上一个模型下降得更快、更低。这显示了学习率在训练过程中的重要性, 我们在未来的项目中应更加关注这一超参数。"
          }
        ]
      },
    ]
  },
  { id: 7, 
    name: "Save a Model",
    sections: [
      { id: 0, 
        name: "5. Save a Model", 
        value: "Before we save the model, we need to choose the best one.",
        image: "",
        content: [
          { id: 1,
            title: "",
            value: "好吧, 作为常规模型训练过程的一部分, 唯一不同的是我们需要选择最佳模型进行保存, 因为我们构建了多个版本的模型以改进原始模型。"
          },
        ]
      },
      { id: 1, 
        name: "5.0 Choose a Best Model", 
        value: "",
        image: "src/assets/chapter_four/save_choose.png",
        content: [
          { id: 1,
            title: "",
            value: "在这里, 我们绘制了所有的模型——从最初的表现不佳的模型到完美的模型_3——以便您清楚地看到它们之间的差异。从模型_0 开始, 您可以看到模型如何逐渐改进, 直到出现完美的模型。这一过程清晰、易于理解且视觉效果显著。"
          }
        ]
      },
      { id: 2, 
        name: "5.1 Save a Model", 
        value: "",
        image: "src/assets/chapter_four/save_model_3.jpeg",
        content: [
          { id: 1,
            title: "",
            value: "按照常规的模型保存过程进行即可, 没有了！"
          }
        ]
      },
      { id: 3, 
        name: "5.2 Load a Model", 
        value: "",
        image: "src/assets/chapter_four/save_load_model_3.jpeg",
        content: [
          { id: 1,
            title: "",
            value: "在第一步——准备数据——我们将数据加载到GPU上 (如果可用) 。然而, 我们将模型保存到CPU上。因此, 在使用保存的模型进行预测之前, 我们需要将模型移动到GPU上 (如果可用) 。这与上一个项目相比是唯一的区别。"
          }
        ]
      },
    ]
  },
]