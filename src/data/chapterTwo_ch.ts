export default [
  { id: 0, 
    name: "PyTorch Workflow",
    sections: [
      { id: 0,
        name: "0. PyTorch 工作流程", 
        value: "A PyTorch workflow typically follows a sequence of steps that guide you through building, training, and evaluating deep learning models.",
        image: "",
        content: [
          { id: 1, 
            title: "",
            value: "PyTorch 工作流程是指使用 PyTorch 开发深度学习模型的一个工作流程。从准备数据开始，然后构建模型、用数据训练模型和评估模型性能。如果模型评估结果不好，还可以微调或重构模型等来提高模型性能，最后我们可以保存高性能模型。这一过程确保了从数据处理到模型部署的系统化流程。"
          },
        ]
      },
      { id: 1,
        name: ":: 工作流程预览", 
        value: "",
        image: "src/assets/chapter_two/workflow.jpeg",
        content: [
          { id: 1, 
            title: "",
            value: "在 PyTorch 机器学习工作流程中，你首先要准备数据，包括收集数据和预处理数据以进行之后的模型训练。接下来，构建模型，定义其架构。然后, 设置训练参数, 并对模型进行训练和评估，以测试其性能并根据需要进行改进。最后，保存模型以备将来使用。"
          },
        ]
      },
      { id: 2,
        name: ":: 工作流程步骤", 
        value: "",
        image: "src/assets/chapter_two/workflow_explain.jpeg",
        content: [
          { id: 1, 
            title: "准备数据",
            value: "我们的第一个项目，是构建一个简单的线性模型，因此数据准备比较简单。我们将生成一组线性数据，并将其分割成训练数据和测试数据。"
          },
          { id: 2, 
            title: "构建模型",
            value: "构建模型主要指设计一个神经网络架构。在这个项目里，我们的模型架构非常简单: 一条直线，由一个线性函数表示，只有两个参数。数据从输入流向输出, 没有隐藏层。"
          },
          { id: 3, 
            title: "训练模型",
            value: "训练和评估模型在此步骤中一同设定。当我们构建训练循环时, 随着参数的更新, 我们会在每次迭代后评估模型的性能。"
          },
          { id: 4, 
            title: "改进模型",
            value: "模型训练完成后, 如果模型评估结果不好，则需要对模型进行微调或者重构。不过，作为我们的第一个模型, 我们将跳过这一步 (模型过于简单, 可以一步到位!)。在之后的所有模型中，我们都将加入此步骤，并重点讨论如何改进模型, 以获得更好的测试结果。"
          },
          { id: 5, 
            title: "保存模型",
            value: "一旦我们训练出一个好的模型，就有必要将其保存起来, 之后是重复使用还是进一步训练都可以直接下载此模型。"
          },
        ]
      },
    ]
  },
  { id: 1, 
    name: "Prepare DATA",
    sections: [
      { id: 0, 
        name: "1. 准备数据", 
        value: "Preparing data is the first step in machine learning, involving the organization and refinement of raw data to ensure it is suitable for training.",
        image: "",
        content: [
          { id: 1,
            title: "",
            value: "准备数据是机器学习中的第一步，也是非常重要的一步。我们需要对原始数据进行必要的处理，以确保其适用于模型训练。数据的质量和结构直接影响模型的性能和准确性，因此这一步对于项目的整体成功至关重要。"
          },
        ]
      },
      { id: 1, 
        name: "1.1 创建数据", 
        value: "",
        image: "src/assets/chapter_two/prepare_create.jpeg",
        content: [
          { id: 1,
            title: "",
            value: "在数据准备阶段，我们首先使用 `torch.arange(0, 1, 0.02)` 创建输入数据，该函数生成了 0 到 1 之间均匀分布的 50 个数字。这些数值将作为我们模型的输入数据。"
          },
          { id: 2,
            title: "",
            value: "为了生成对应的输出数据，我们使用一个简单的线性方程 (y = 0.6x + 0.1) 作为我们模型的目标值。"
          }
        ]
      },
      { id: 2, 
        name: "1.2 分割数据", 
        value: "",
        image: "src/assets/chapter_two/prepare_split.jpeg",
        content: [
          { id: 1,
            title: "",
            value: "生成输入数据和输出数据后，下一步是将数据拆分为训练数据集和测试数据集。我们将 40 个数据集合（总数的 80%）分配给训练数据集，模型将使用这些数据来学习输入和输出之间的关系。"
          },
          { id: 2,
            title: "",
            value: "剩余的 10 个数据点（总数的 20%）则保留为测试数据集，以便我们评估模型在未知数据上的表现。这种拆分确保了模型的有效训练，同时也提供了一种评估其准确性和泛化能力的方法。"
          }
        ]
      },
      { id: 3, 
        name: ":: 数据可视化 ", 
        value: "",
        image: "src/assets/chapter_two/prepare_visual.png",
        content: [
          { id: 1,
            title: "",
            value: "在这里，我们使用 matplotlib 的 pyplot 模块可视化我们的目标模型。生成的图像清晰地显示了我们的模型是呈线性分布的，其中训练数据标记为蓝色，测试数据标记为绿色。"
          },
          { id: 2,
            title: "",
            value: "这种可视化方法提供了一种清晰有效的方式来展示数据，使我们更容易理解模型的结构。在接下来的步骤和以后的项目中，我们将广泛使用这个可视化工具。"
          }
        ]
      },
    ]
  },
  { id: 2, 
    name: "Build a Model",
    sections: [
      { id: 0, 
        name: "2. 创建一个模型", 
        value: "Building a model involves designing and implementing a machine learning algorithm that learns patterns from data to make predictions or decisions.",
        image: "",
        content: [
          { id: 1,
            title: "",
            value: "构建模型是机器学习过程中主要步骤，在这个阶段，我们需要设计并实现一个能够从数据中学习特定模式的深度学习学习模型。这个步骤包括选择最适合当前问题的模型架构。"
          },
          { id: 2,
            title: "",
            value: "一旦模型架构确定，就可以使用像 PyTorch 这样的框架对其进行实现。这涉及到设置模型的层、指定激活函数和初始化权重等等。之后，模型就准备好在数据集上进行训练，在训练过程中。模型会自主调整其参数, 以最小化损失值来提高模型预测的准确性。"
          },
        ]
      },
      { id: 1, 
        name: ":: 创建一个模型类", 
        value: "",
        image: "src/assets/chapter_two/build_model.jpeg",
        content: [
          { id: 1,
            title: "",
            value: "在构建模型时，第一步是定义和初始化模型将用来从数据中学习的参数。这些参数，如权重和偏置，是至关重要的，因为它们决定了模型如何处理输入数据。"
          },
          { id: 2,
            title: "",
            value: "一旦参数设置完成，我们接下来可以定义计算过程，描述数据如何在模型中流动，从输入到输出，以及模型如何使用这些参数来进行预测。"
          }
        ]
      },
      { id: 2, 
        name: ":: 模型架构", 
        value: "",
        image: "src/assets/chapter_two/build_architecture.jpeg",
        content: [
          { id: 1,
            title: "",
            value: "这个模型的结构非常简单，仅仅一个输入和一个输出，且通过一个线性函数连接。"
          }
        ]
      },
      { id: 3, 
        name: ":: 模型可视化", 
        value: "",
        image: "src/assets/chapter_two/build_visual.png",
        content: [
          { id: 1,
            title: "",
            value: "我们使用红色圆点绘制了初始模型在训练数据集上的预测结果，并将其与目标模型（绿色圆点）进行比较。这使我们能够清楚地知道当前模型与目标模型之间的差异。"
          }
        ]
      },
    ]
  },
  { id: 3, 
    name: "Train a Model",
    sections: [
      { id: 0, 
        name: "3. 训练模型", 
        value: "Training a model involves optimizing its parameters using labeled data, while testing evaluates the model’s performance on unseen data to assess its generalization ability.",
        image: "",
        content: [
          { id: 1,
            title: "",
            value: "训练模型的时候, 我们主要根据标记数据来调整模型参数，以最小化损失来提高准确性。测试则是用于评估模型在新数据集上的表现，以衡量其对不同场景的泛化能力。"
          },
        ]
      },
      { id: 1, 
        name: "3.1 创建一个训练循环", 
        value: "",
        image: "src/assets/chapter_two/train_model.jpeg",
        content: [
          { id: 1,
            title: "选择损失函数",
            value: "这是模型训练中的第一个步骤，损失函数量化了模型预测与实际标签的匹配程度。"
          },
          { id: 2,
            title: "选择 Optimizer",
            value: "Optimizer 负责根据损失函数的梯度更新模型的参数。选择合适的优化器（如SGD、Adam等）会影响模型的收敛性和性能。"
          },
          { id: 3,
            title: "设置循环次数",
            value: "设定合适的模型训练次数 (每次循环需要跑完整个数据集) 很重要，这样可以确保模型有足够的时间进行学习，但又不会过度学习造成过拟合。这一步涉及平衡训练时间和模型性能。"
          },
          { id: 4,
            title: "设置训练步骤",
            value: "训练循环中的每一个步骤都是模型实际学习的必要条件。在每次迭代中, 模型批次处理输入数据，计算损失，并使用优化器 (Optimizer) 更新参数。"
          },
          { id: 5,
            title: "设置测试步骤",
            value: "每一次训练迭代之后, 我们都会用测试数据集对当前模型进行评估, 以此确认模型的泛化能力。"
          },
        ]
      },
      { id: 2, 
        name: ":: 可视化训练模型", 
        value: "",
        image: "src/assets/chapter_two/train_visual_before.jpeg",
        content: [
          { id: 1,
            title: "",
            value: "我们已经绘制了初始模型，现在让我们重点了解如何训练它。模型训练过程其实非常简单, 它主要包括两个关键步骤:计算损失, 然后优化。基本上，我们都在重复这两个步骤, 直到找到模型的最佳参数。"
          }
        ]
      },
      { id: 3, 
        name: ":: 可视化训练后的模型", 
        value: "",
        image: "src/assets/chapter_two/train_visual_after.jpeg",
        content: [
          { id: 1,
            title: "",
            value: "在完成100个训练循环之后, 我们再次绘制了我们的模型。从图像中可以看出，我们的模型与目标模型之间的差异现在已经非常小了很多。与初始模型相比，我们当前的模型更接近目标。"
          }
        ]
      },
      { id: 4, 
        name: ":: 测试模型", 
        value: "",
        image: "src/assets/chapter_two/train_visual_testd.jpeg",
        content: [
          { id: 1,
            title: "",
            value: "我们使用测试数据对模型进行了评估，并绘制了图像。从图像中可以看出，这些红点与代表目标值的绿点已经非常接近了。"
          }
        ]
      },
      { id: 5, 
        name: ":: 确认损失曲线", 
        value: "损失曲线显示了模型在训练过程中损失数值的变化，这两条训练数据下的损失曲线和测试数据下的损失曲线。图中可以看出, 这两条曲线下降趋势明显且最终都收敛, 这表明模型在学习和泛化方面表现良好。",
        image: "src/assets/chapter_two/train_loss_curves.jpeg",
        content: [
          { id: 1,
            title: "训练损失值 (蓝色)",
            value: "训练损失随着训练次数的增加而稳步下降，表明模型在训练数据上的学习效果良好。曲线在接近结束时趋于平稳，暗示模型趋于收敛。"
          },
          { id: 2,
            title: "测试损失值 (橙色)",
            value: "测试损失也随着时间的推移而减少，但起始值高于训练损失值，并且在整个训练过程中始终高于训练损失。这表明模型在泛化方面表现合理，但在未见过的测试数据上的表现略逊于训练数据。"
          },
          { id: 3,
            title: "归纳",
            value: "两条曲线的平缓下降是一个好的标志，表明模型在训练过程中在不断优化，并且没有出现过拟合，因为测试损失也在随着时间降低。进一步的调整可以缩小这两条曲线之间的差距, 这里我们不再讨论。"
          }
        ]
      },
    ]
  },
  { id: 4, 
    name: "Save a Model",
    sections: [
      { id: 0, 
        name: "4. Save a Model", 
        value: "Saving a model involves storing the trained model’s parameters to a file, allowing you to reuse the model later without retraining it from scratch.",
        image: "",
        content: [
          { id: 1,
            title: "",
            value: "保存模型是机器学习工作流程中的最后一步。在训练模型之后，你通常需要保存一个优秀的模型，方便以后需要部署、共享，货值继续训练。"
          },
          { id: 2,
            title: "",
            value: "保存的模型包括学习到的参数（权重和偏置），有时还包括模型的架构，这样可以方便我们随时下载模型并用于预测。"
          },
        ]
      },
      { id: 1, 
        name: "4.1 保存模型", 
        value: "",
        image: "src/assets/chapter_two/save_model.jpeg",
        content: [
          { id: 1,
            title: "",
            value: "首先，我们需要创建一个目录，并在该目录下保存模型的参数。"
          }
        ]
      },
      { id: 2, 
        name: "4.2 下载模型", 
        value: "",
        image: "src/assets/chapter_two/save_model_load.jpeg",
        content: [
          { id: 1,
            title: "",
            value: "之后，当我们需要使用这个模型时，只需要下载该模型的参数到一个初始化模型中就可以了。"
          }
        ]
      },
    ]
  },
]