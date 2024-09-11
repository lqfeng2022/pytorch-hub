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
            value: "PyTorch 工作流是使用 PyTorch 开发深度学习模型的结构化方法。它从准备数据开始，然后涉及构建模型、用数据训练模型以及评估模型性能。在评估模型后，你可以进行改进，最后保存模型以备后用。这一过程确保了从数据处理到模型部署的系统化流程。"
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
            value: "在 PyTorch 机器学习工作流中，你首先要准备数据，包括收集和预处理数据以进行模型训练。接下来，构建模型，定义其架构并设置训练参数。然后对模型进行训练和评估，以测量其性能并根据需要进行改进。最后，保存模型并加载以备将来使用，从而完成工作流。"
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
            value: "在我们的第一个项目中，我们将处理一个简单的直线，因此数据准备比较简单。我们将生成线性数据，并将其分成训练集和测试集。"
          },
          { id: 2, 
            title: "创建模型",
            value: "构建模型涉及设计神经网络架构。在这个初始项目中，我们的模型非常简单——一个直线，由一个线性函数表示，只有两个参数。没有隐藏层；数据直接从输入流向输出。"
          },
          { id: 3, 
            title: "训练模型",
            value: "训练和评估模型在此步骤中结合进行。当我们构建训练循环时，还会在每次迭代后评估模型的性能，随着参数的更新。这个过程将遵循训练和测试神经网络的典型步骤。"
          },
          { id: 4, 
            title: "改进模型",
            value: "训练后，评估模型的性能至关重要。如果结果不令人满意，则需要进行改进。不过，在我们的第一个模型中，我们将跳过这一步。在未来的项目中，我们将深入探讨这个过程，重点关注改进模型以获得更好和更高效的结果"
          },
          { id: 5, 
            title: "保存模型",
            value: "一旦我们开发出一个好的模型，就有必要将其保存以备重用或进一步训练。在这种情况下，我们主要保存模型的参数。"
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
            value: "准备数据是机器学习中的第一步，涉及组织和精炼原始数据，以确保其适合用于训练。数据的质量和结构直接影响模型的性能和准确性，因此这一步对于项目的整体成功至关重要。"
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
            value: "在数据准备阶段，我们首先使用 `torch.arange(0, 1, 0.02)` 创建输入数据，该函数生成了 0 到 1 之间均匀分布的 50 个数字。这些数值将作为我们模型的输入。"
          },
          { id: 2,
            title: "",
            value: "为了生成对应的输出数据，我们应用一个简单的线性公式：`y = 0.6x + 0.1`。这个方程产生了我们的模型将学习预测的目标值，从而在输入和输出之间建立了一个明确的线性关系。"
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
            value: "生成输入和输出数据后，下一步是将数据拆分为训练集和测试集。我们将 40 个数据点（总数的 80%）分配给训练集，模型将使用这些数据来学习输入和输出之间的关系。"
          },
          { id: 2,
            title: "",
            value: "剩余的 10 个数据点（总数的 20%）保留为测试集，以便我们评估模型在未见过的数据上的表现。这种拆分确保了模型的有效训练，同时也提供了一种评估其准确性和泛化能力的方法。"
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
            value: "在这里，我们使用 matplotlib 的 pyplot 模块可视化我们的目标模型。生成的图像清晰地显示了我们的模型作为一系列排列成直线的点，其中训练数据标记为蓝色，测试数据标记为绿色。"
          },
          { id: 2,
            title: "",
            value: "这种可视化方法提供了一种清晰有效的方式来展示数据，使我们更容易理解模型的结构。在接下来的步骤和未来的项目中，我们将频繁使用这种方法。"
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
            value: "构建模型是机器学习过程中至关重要的步骤，在这个阶段，我们设计并实现一个能够从数据中学习模式的机器学习算法。这个步骤包括选择最适合当前问题的模型架构。"
          },
          { id: 2,
            title: "",
            value: "一旦架构确定，模型就可以使用像 PyTorch 这样的框架进行实现。这涉及到设置模型的层、指定激活函数和初始化权重。接着，模型就准备好在数据上进行训练，在训练过程中，它会调整其参数以最小化错误并改善预测结果。"
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
            value: "这个模型的结构非常简单，仅由一个输入和一个输出组成，中间通过一个线性函数连接。"
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
            value: "我们使用红色点绘制了初始模型，并将其与目标模型（绿色点）进行比较。这使我们能够清楚地看到当前模型与目标模型之间的差异。如预期的那样，差异非常明显。"
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
            value: "训练模型涉及根据标记数据调整其参数，以最小化错误并提高准确性。测试则评估模型在新数据上的表现，以衡量其对不同场景的泛化能力。"
          },
        ]
      },
      { id: 1, 
        name: "3.1 创建一个训练循环", 
        value: "",
        image: "src/assets/chapter_two/train_model.jpeg",
        content: [
          { id: 1,
            title: "Pick Up a Loss Function",
            value: "这是模型训练中的第一个关键步骤，因为损失函数量化了模型预测与实际标签的匹配程度。它作为模型需要最小化的目标函数。"
          },
          { id: 2,
            title: "Pick Up an Optimizer",
            value: "优化器负责根据损失函数的梯度更新模型的参数。选择合适的优化器（如SGD、Adam等）可以显著影响模型的收敛性和性能。"
          },
          { id: 3,
            title: "Set Up Training Epochs",
            value: "定义轮数（对整个数据集的迭代次数）很重要，以确保模型有足够的时间进行学习，但又不会过度拟合。这一步涉及平衡训练时间和模型性能。"
          },
          { id: 4,
            title: "Set Up Training Loop",
            value: "训练循环是实际学习发生的地方。它迭代地将数据批次输入模型，计算损失，并使用优化器更新参数。这一步实现了训练过程的核心。"
          },
          { id: 5,
            title: "Set Up Testing Loop",
            value: "训练完成后，测试循环对模型在未见数据上的表现进行评估。它通过检查模型在单独测试集上的性能来帮助评估模型的泛化能力。"
          },
        ]
      },
      { id: 2, 
        name: ":: 可视化初始模型", 
        value: "",
        image: "src/assets/chapter_two/train_visual_before.jpeg",
        content: [
          { id: 1,
            title: "",
            value: "我们已经绘制了初始模型，现在让我们重点了解如何训练它。训练过程其实非常简单。主要包括两个关键步骤：计算损失，然后使用优化器。基本上，我们重复这两个步骤，直到找到模型的最佳参数。"
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
            value: "在完成100个训练轮次后，我们再次绘制我们的模型。从图像中可以看出，我们的模型与目标模型之间的差异现在已经非常小。与初始模型相比，我们当前的模型更接近目标。"
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
            value: "我们使用测试数据对训练好的模型进行预测，并将结果绘制为红点。从图像中可以看出，这些红点与代表目标值的绿点非常接近。"
          }
        ]
      },
      { id: 5, 
        name: ":: 确认损失曲线", 
        value: "损失曲线显示了模型在训练过程中错误的减少情况，包括训练数据和测试数据的单独曲线。理想情况下，这两条曲线都应显示下降趋势，表明模型在学习和泛化方面表现良好。",
        image: "src/assets/chapter_two/train_loss_curves.jpeg",
        content: [
          { id: 1,
            title: "训练损失值 (蓝色)",
            value: "训练损失随着训练轮次的增加而稳步下降，表明模型在学习并提升其在训练数据上的表现。曲线在接近结束时趋于平稳，暗示模型接近收敛。"
          },
          { id: 2,
            title: "测试损失值 (橙色)",
            value: "测试损失也随着时间的推移而减少，但起始值高于训练损失，并且在整个训练过程中始终高于训练损失。这表明模型在泛化方面表现合理，但在未见过的测试数据上的表现略逊于训练数据。"
          },
          { id: 3,
            title: "Overall",
            value: "两条曲线的下降是一个好兆头，表明模型在训练过程中不断改善，并且没有过拟合，因为测试损失也在随着时间降低。进一步的调整可以缩小这两条曲线之间的差距。"
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
            value: "S保存模型是机器学习工作流中的一个关键步骤。在训练模型之后，您通常会希望保存它，以便可以部署、共享，或在以后继续训练，而无需从头开始。"
          },
          { id: 2,
            title: "",
            value: "保存的模型包括学习到的参数（权重和偏置），有时还包括模型的架构，使得能够重新加载并立即用于预测。"
          },
          { id: 3,
            title: "",
            value: "保存模型确保了训练过程中花费的时间和资源不会丢失，并且使得在生产环境中部署模型变得更加容易，模型可以用来对新数据进行预测。此外，保存的模型可以与他人共享或在不同的项目中重用。"
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
            value: "首先，我们为我们的模型创建一个目录，并使用 PyTorch 的 save() 方法保存其状态 dictionary 。"
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
            value: "之后，当我们需要使用这个模型时，只需使用 load_state_dict() 方法将状态 dictionary 加载回模型即可。"
          }
        ]
      },
    ]
  },
]