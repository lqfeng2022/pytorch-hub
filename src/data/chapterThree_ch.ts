export default [
  { id: 0, 
    name: "Linear Regression",
    sections: [
      { id: 0,
        name: "1. 线性回归", 
        value: "Linear regression is a statistic method that models a straight-line relationship between 2 types of variables.",
        image: "",
        content: [
          { id: 1, 
            title: "直线关系",
            value: "线性回归是一种统计方法, 用于建立变量间的一种线性模型。通过确定一条直线, 来最小化预测值和观测值之间的差异。这种方法可用在很多领域, 用来预测或者分析某中趋势。"
          },
          { id: 2, 
            title: "机器学习中的线性回归",
            value: "线性回归是机器学习中最基本的算法, 也是机器学习最重要的一个算法, 没有比一条直线更简单的了, 学习线性回归是理解深度学习模型的第一步。在机器学习中, 线性回归多用于预测持续变化的数据, 例如价格、温度等。"
          },
        ]
      },
      { id: 1,
        name: ":: 线性关系", 
        value: "",
        image: "src/assets/chapter_three/linear.jpeg",
        content: [
          { id: 1, 
            title: "",
            value: "图中直线是一个用于预测工作时间和薪水之间关系的线性模型。这个模型显示了, 每多工作一个小时, 薪水会增加 $10, 一个简单的线性回归问题。该模型准确地捕捉并预测了这两个变量之间的关系。"
          },
        ]
      },
    ]
  },
  { id: 1, 
    name: "Normal Distribution",
    sections: [
      { id: 0,
        name: "2. 正态分布", 
        value: "A Normal Distribution is a type of continuous probability distribution for a real-valued random variable, typically forming a bell curve.",
        image: "",
        content: [
          { id: 1, 
            title: "什么是正态分布？",
            value: "正态分布是一种用于描述所观测值 (比如年龄, 财富值等等) 随机分布的概率曲线。一句话来说, 它是一种描述观测值自然分布方式的方法。当观测值符合正态分布时, 大多数会集中在均值 (中心点) 附近, 随着偏离中心点越远, 观测值的数量会逐渐减少。这形成了经典的“钟形曲线”, 该曲线在两侧对称。"
          },
          { id: 2, 
            title: "为什么我们要了解正态分布？",
            value: "正态分布是机器学习和深度学习的基石。在我们构建第一个模型时, 我们使用正态分布来初始化权重和偏差。这有助于模型以平稳的方式开始训练, 防止出现梯度突然消失或指数增长的问题。更多的细节我们会在梯度下降中说明。"
          },
        ]
      },
      { id: 1,
        name: "2.1 概率密度函数 (PDF)", 
        value: "让我们了解下概率密度函数, 即 PDF, 当我门说到正态分布时, 一般指的就是它。我们不需要去记住其公式, 记住其形状, 理解参数是如何影响其形状和位置的就可以了。",
        image: "src/assets/chapter_three/ndistrib_pdf.jpeg",
        content: [
          { id: 1, 
            title: "钟形曲线:",
            value: "想象一个钟, 个人觉得想象成山行会更有意思。"
          },
          { id: 2, 
            title: "均值 (µ = 0):",
            value: "就是钟的顶点, 或者说山顶。在标准正态分布中, µ 等于 0。"
          },
          { id: 3, 
            title: "标准差 (σ = 1):",
            value: "它决定了钟的宽度。在标准正态分布中, σ 等于 1。"
          },
          { id: 4, 
            title: "x轴",
            value: "代表随机变量 (或者说观测值) 的可能值。"
          },
          { id: 5, 
            title: "y轴:",
            value: "概率密度, 或者说 x轴 上每个观测值的可能性。"
          },
        ]
      },
      { id: 2,
        name: "2.2 累积分布函数 (CDF)", 
        value: "累积分布函数 (CDF) 可以通过 PDF曲线 下方特定点(我们称之为 x) 左侧的阴影区域来可视化。它表示随机变量 X 取值小于或等于 x 的概率。",
        image: "src/assets/chapter_three/ndistrib_cdf.jpeg",
        content: [
          { id: 1, 
            title: "S形曲线:",
            value: "一个典型的 S 形曲线, 是山形曲线的积分所得。"
          },
          { id: 2, 
            title: "一个特殊点 - (µ, 0.5):",
            value: "在标准正态分布中, x = 0 时的 CDF 为 0.5。这意味着变量落在均值左侧的概率是 50%。"
          },
          { id: 3, 
            title: "x轴",
            value: "代表随机变量的值。"
          },
          { id: 4, 
            title: "y轴:",
            value: "代表累积概率。"
          },
        ]
      },
    ]
  },
  { id: 2,
    name: "Loss Function",
    sections: [
      { id: 0,
        name: "3. 损失函数",
        value: "The Loss Function is used to evaluate how well your model’s predictions are performing, the lower the value, the better the model is doing.",
        image: "",
        content: [
          { id: 1,
            title: "什么是损失函数",
            value: "损失函数在评估模型预测数值与实际数值的匹配程度中至关重要。它衡量模型预测值与真实值之间的差异, 目标是将这种差异最小化。简单来说, 损失越低, 说明模型的预测越准确。"
          },
          { id: 2,
            title: "工作原理",
            value: "损失函数计算模型预测值与数据集中实际值之间的差异。这种差异被汇总为一个标量值, 表示“损失”或误差。"
          },
          { id: 3,
            title: "为什么它很重要",
            value: "损失函数在优化过程中发挥着重要作用。在训练过程中, 模型利用损失值来微调其内部参数(如权重和偏差), 以减少损失, 从而随着时间的推移提供更好的预测。"
          },
          { id: 4,
            title: "损失值越低越好",
            value: "训练中的关键目标是最小化损失函数。较低的损失表示模型的预测值更接近实际值, 说明模型正在有效学习。"
          }
        ]
      },
      { id: 1,
        name: "3.1 均方误差 (MSE)",
        value: "Mean Squared Error, or MSE for short, is a type of loss function. It’s quite sensitive to outliers because larger errors are squared, making them more significant.",
        image: "",
        content: [
          { id: 1,
            title: "",
            value: "均方误差计算预测值与实际值之间差异的平方的平均值。由于较大的误差被平方, 因此MSE对误差更为敏感, 使其显得更为重要。"
          },
        ]
      },
      { id: 2,
        name: ":: 直线模型中的MSE(第1部分) ",
        value: "",
        image: "src/assets/chapter_three/mse_one.jpeg",
        content: [
          { id: 1,
            title: "",
            value: "该图显示了两个线性模型：红色代表当前模型的预测值, 蓝色代表目标模型的真实值。当前线性模型由方程  y = 0.3x + b  表示, 其中b是偏差。目标模型的线性方程为  Y = 0.6x + 0.1 。预测值与目标值之间的差异通过垂直箭头表示, 显示了两者之间的差距。"
          },
        ]
      },
      { id: 3,
        name: ":: 直线模型中的MSE(第2部分) ",
        value: "",
        image: "src/assets/chapter_three/mse_two.jpeg",
        content: [
          { id: 1,
            title: "MSE计算:",
            value: "MSE损失函数通过预测值与实际目标值之间的平方差的平均值来计算。这个公式衡量了预测值与实际值之间的偏离程度。"
          },
          { id: 2, 
            title: "MSE图:",
            value: "右侧的图表示了MSE损失函数与偏差b的函数关系。这条曲线显示了当b变化时MSE如何变化, 损失函数的最小值即曲线的最低点, 则表示了b的最佳取值点。"
          },
          { id: 3, 
            title: "目标:",
            value: "训练模型的目标是调整偏差b(从考虑一个变量开始)以最小化MSE损失函数。这可以通过找到MSE曲线上的最低点来直观地表示。"
          },
        ]
      },
      { id: 4,
        name: "3.2 损失曲线",
        value: "The Loss Curve is a graph that tracks a model’s error or losses over the course of training.",
        image: "",
        content: [
          { id: 1,
            title: "",
            value: "损失曲线是一个图形工具, 用于跟踪模型在训练期间的错误或损失。它帮助我们直观告诉了我们模型在训练和评估过程中周期性变化。通常, 我们用两条曲线来共同表示: 训练损失曲线显示训练数据上的损失如何随epoch减少, 而测试损失曲线则展示了测试数据上的损失在相同时间段内的变化。"
          },
        ]
      },
      { id: 5,
        name: ":: 三种损失曲线",
        value: "",
        image: "src/assets/chapter_three/losscurves.jpg",
        content: [
          { id: 1,
            title: "欠拟合(Underfitting) :",
            value: "最左边的曲线图为“欠拟合”, 训练损失值(蓝线)和测试损失值(红线)都很高且下降缓慢。两者之间的差距很小, 但都可以更低。这说明该模型没有学习到数据中的潜在模式。"
          },
          { id: 2,
            title: "过拟合(Overfitting) :",
            value: "中间的曲线图为“过拟合”, 显示了训练损失值(蓝线)迅速下降且至极低点, 而测试损失值(红线)下降到中途就停滞不前且有上升趋势。这意味着模型在训练数据上表现非常好, 但在测试数据上表现较差, 说明模型对训练集过于依赖, 没有得到很好的泛化。"
          },
          { id: 3,
            title: "恰到好处(Just Right) :",
            value: "最右边的曲线图为“恰到好处”, 训练损失(蓝线)和测试损失(红线)都平稳下降并趋于相似的较低值。这表明模型调整得很好, 从训练数据中取得了好的学习效果, 同时也能够很好地泛化至测试数据。"
          }
        ]
      },
    ]
  },
  { id: 3,
    name: "Gradient Descent",
    sections: [
      { id: 0,
        name: "4. 梯度下降",
        value: "Gradient Descent is an optimization algorithm that calculates the gradient (slope) using all the samples in the training set to update the model’s parameters during each iteration.",
        image: "",
        content: [
          { id: 1,
            title: "",
            value: "想象一下, 周末你和朋友去爬山, 结果在途中迷路了。茂密的树林遮住了视线，你们看不到下山的路。而且太阳很快就要落山了，时间紧迫，你们得想办法尽快找到下山的路。这个寻找路径的过程，就像梯度下降一样。山的地形就好比损失函数，地形的起伏变化就代表了梯度。当你们寻找下山的路径时，地形无论是陡峭还是平缓，总是朝着更低的地方延伸，直到最终走到山脚，也就是所谓的全局最优点。"
          },
        ]
      },
      { id: 1,
        name: "4.1 梯度下降模拟(1个参数) ",
        value: "",
        image: "src/assets/chapter_three/gd_one.jpeg",
        content: [
          { id: 0,
            title: "",
            value: "该图直观地展示了梯度下降 (Gradient Descent) 如何通过迭代调整模型参数，逐步缩小模型预测值（红点）与实际值（蓝点）之间的误差。每一步中，模型的预测直线从 y = 0.3x 向目标直线 y = 0.6x + 0.1 靠拢，有效降低了整体误差。随着模型直线逐步逼近目标直线，清晰展示了梯度下降在提升模型性能上的有效性。"
          },
          { id: 1,
            title: "初始直线(红色) :",
            value: "红点表示模型在参数 b = 0 时做出的初始预测结果。红色线条则代表模型在进行梯度下降迭代前的起始状态。"
          },
          { id: 2,
            title: "目标直线(蓝色) :",
            value: "蓝点表示模型应当预测的理想目标值，而穿过这些蓝点的蓝色线条则是模型通过梯度下降希望逼近的最优拟合线。"
          },
          { id: 3,
            title: "梯度下降迭代:",
            value: "红色线条与蓝色线条之间的灰色线条展示了每次梯度下降迭代后模型预测线的变化。每次更新都会缩小模型预测值与目标值之间的差距，逐步将预测线条向目标线条靠拢。这个过程直观地演示了模型在迭代中逐步收敛到更准确预测值的过程。"
          },
        ]
      },
      { id: 2,
        name: ":: 梯度下降模拟(表格)",
        value: "",
        image: "src/assets/chapter_three/gd_one_table.jpeg",
        content: [
          { id: 1,
            title: "",
            value: "在这个表格中, 我们展示了参数b在多次迭代中的变化, 以及对应的均方误差(MSE) 和其导数。算法从初始值 b = 0 开始 (为了方便演示), 并根据该点的 MSE 梯度(斜率) 逐步更新b。更新规则遵循以下公式：b(n+1) = b(n) - lr * MSE’(b), 其中lr表示学习率, 我们将其设置为 0.2。"
          },
        ]
      },
      { id: 3,
        name: ":: MSE的梯度下降",
        value: "",
        image: "src/assets/chapter_three/gd_one_mse.jpeg",
        content: [
          { id: 1,
            title: "MSE曲线",
            value: "左侧的图展示了 MSE 损失函数随着参数 b 变化的曲线，曲线表明我们调整 b 时误差的变化情况。曲线上的红点显示了梯度下降的进程，从初始值 b = 0 开始，经过每次更新后， b 逐步向更低的 MSE 值移动。与曲线相切的线表示每个点的梯度，指引更新的方向和步幅。"
          },
          { id: 2,
            title: "MSE的导数",
            value: "右侧的图展示了 MSE 关于参数 b 的导数。红点表示每次迭代中梯度的数值。随着算法接近最小值，梯度逐渐减小。当 b 接近最小化 MSE 的值时，梯度趋近于零。当梯度为零时，算法表明已达到 b 的最优值。"
          },
        ]
      },
      { id: 4,
        name: "4.2 梯度下降 (2个参数)",
        value: "这张图生动地展现了两个参数下的梯度下降过程。通过3D曲面图，我们可以直观地看到模型如何像在崎岖的山谷中寻找下坡的路径，一步步朝着误差最小的方向前进。",
        image: "src/assets/gradient_descent.jpeg",
        content: [
          { id: 1,
            title: "损失函数曲面图:",
            value: "图中展示了两个参数的损失函数曲面图，曲面的高度代表了在不同 θ1 和 θ2 取值下损失函数的大小。梯度下降的目标是沿着这个曲面找到损失函数的最低点，也就是整个曲面上的全局最小值。"
          },
          { id: 2,
            title: "梯度下降过程:",
            value: "曲面上的箭头指示了梯度下降算法在调整 θ1 和 θ2 时的路径，随着每次迭代，算法逐步接近最小损失值。这些箭头清晰地展示了算法如何顺着曲面的斜坡（梯度）一路向下，朝着最低点前进。"
          },
          { id: 3,
            title: "损失函数的等高线:",
            value: "图中还展示了损失函数的等高线，深色区域代表较高的损失值，而浅色区域则表示较低的损失值。梯度箭头从深色区域逐渐移动到浅色区域，直观地展现了损失值不断优化的过程。"
          },
          { id: 4,
            title: "坐标轴:",
            value: "x轴和y轴表示参数θ1和θ2, z轴为损失函数。"
          },
        ]
      },
      { id: 5,
        name: ":: 梯度下降 (表格)",
        value: "",
        image: "src/assets/chapter_three/gd_two_table.jpeg",
        content: [
          { id: 1,
            title: "",
            value: "在这个表格中，我们计算了两个参数（偏置 b 和权重 w) 的数值变化，以及对应的均方误差 (MSE) 和其导数，展示了在多次迭代中的变化过程。这里的初始值设置为 b = 0, w = 0.3。"
          },
          { id: 2,
            title: "",
            value: "接下来，我们根据 MSE 的梯度来更新 w 和 b，更新规则与单一参数的损失函数类似。需要注意的是，在计算一个参数的梯度时，另一个参数被视为常量。"
          },
        ]
      },
      { id: 6,
        name: ":: 梯度下降 (可视化)",
        value: "",
        image: "src/assets/chapter_three/gd_two.jpeg",
        content: [
          { id: 1,
            title: "",
            value: "这张图展示了梯度下降如何通过迭代调整线性模型的两个参数：权重 (w) 和偏置 (b)，以优化模型的表现。"
          },
          { id: 2,
            title: "",
            value: "之前，我们只考虑了一个参数 (偏置 b) 的损失函数，并将权重固定在 w = 0.3。这使我们能够观察到梯度下降如何调整偏置以减少误差。现在，我们将同时考虑这两个参数，共同参与模型的优化过程。"
          },
          { id: 3,
            title: "",
            value: "当这两个参数都在损失函数中时，梯度下降算法在每次迭代中会同时更新它们，逐步减少模型的误差。这与只考虑一个参数的损失函数的迭代过程类似。唯一的区别在于，这次我们需要同时计算 w 和 b 的梯度，并据此调整模型。"
          },
          { id: 4,
            title: "",
            value: "这张图形象地展示了模型在每次迭代中的变化。每次迭代后，参数更新使当前模型逐步接近目标。虽然我们在这里加入了一个额外的变量，增加了计算复杂度，但与之前单一变量的情况类似，基本原理保持不变。"
          },
        ]
      },
    ]
  },
  { id: 4,
    name: "Stochastic Gradient Descent",
    sections: [
      { id: 0,
        name: "5. 随机梯度下降 (SGD)",
        value: "Stochastic Gradient Descent (SGD) works by using a single sample or a randomly selected subset of the data to perform the same process as regular Gradient Descent.",
        image: "",
        content: [
          { id: 1,
            title: "什么是随机梯度下降(SGD) ",
            value: "随机梯度下降（SGD）是一种机器学习中的优化算法。与常规的梯度下降不同，SGD 在每次迭代时只使用一个样本或一小部分随机样本来更新模型参数，而不是使用整个数据集。这样可以使损失函数更快地收敛，但也会引入较大的波动，从而带来一些不稳定性。"
          },
          { id: 2,
            title: "为什么使用 SGD",
            value: "相比常规梯度下降，SGD 更快且节省内存，尤其适用于大规模数据集。由于每次只使用一个或一小部分数据进行更新，SGD 能更快速、频繁地调整参数。然而，由于使用的是较小的数据集，优化路径可能会更加崎岖，收敛过程也不如梯度下降那样平滑和稳定。"
          },
        ]
      },
      { id: 1,
        name: ":: 随机梯度下降与梯度下降 (part.1)",
        value: "",
        image: "src/assets/chapter_three/sgd_one.jpeg",
        content: [
          { id: 1,
            title: "大数据集",
            value: "梯度下降（GD）在处理大型数据集时效率较低，因为它每次更新都需要遍历整个数据集来计算损失函数。而随机梯度下降（SGD）则不同，它每次只使用一个样本或一小部分样本来更新参数，因此即使面对大数据集，也能快速计算出损失值。"
          },
          { id: 2,
            title: "大量参数:",
            value: "GD 在处理大量参数时表现良好，但由于需要同时计算所有参数的梯度，速度可能较慢。而 SGD 因为更新频率更高，在处理大量参数时具备更好的扩展性。"
          },
          { id: 3,
            title: "参数更新:",
            value: "GD 在遍历完整个数据集后才更新参数，虽然更新稳定且准确，但过程较慢。相反，SGD 每次根据一个样本或一小部分样本来更新参数，虽然速度更快，但也带来了一定的噪声。"
          },
          { id: 4,
            title: "RAM 使用:",
            value: "GD 每次都需要处理整个数据集，因此计算内存需求较大。而 SGD 只处理少量数据，对内存的压力小得多。"
          },
          { id: 5,
            title: "收敛速度:",
            value: "GD 的收敛速度较慢，但过程稳定，能够更加精确地接近最小值。相比之下，SGD 收敛更快，但由于更新过程中的波动，可能会在最小值附近发生振荡。"
          },
          { id: 6,
            title: "准确性:",
            value: "GD 通常可以达到较高的准确性，因为更新过程中的噪声较少，能够更好地反映数据的整体趋势。而 SGD 由于伴随一定噪声，准确性略低，但可以通过学习率衰减等技术来改善。"
          },
        ]
      },
      { id: 2,
        name: ":: 随机梯度下降与梯度下降 (part.2)",
        value: "这幅图清晰地展示了梯度下降（GD）和随机梯度下降（SGD）在模型训练中的差异。",
        image: "src/assets/chapter_three/sgd_two.jpeg",
        content: [
          { id: 1,
            title: "梯度下降(GD) :",
            value: "从图中可以看到，梯度下降在朝向最优点的过程中步伐非常稳定，但在处理大数据集时，速度变得非常缓慢，且计算成本极高。"
          },
          { id: 2,
            title: "随机梯度下降(SGD) :",
            value: "随机梯度下降提供了一种更快速且节省内存的替代方案。通过 SGD，我们可以迅速接近最优点，虽然步伐有些飘忽，专业术语称为噪声，这使得路径变得不太稳定且不可预测。不过，这种噪声有助于跳出局部最小值（可参考梯度下降部分的二维曲面图）。此时，我们可以通过学习率递减等技术来稳定最终的收敛过程。"
          },
        ]
      },
    ]
  },
  { id: 5,
    name: "",
    sections: [
      { id: 0,
        name: "6. 学习率 (lr)",
        value: "Learning Rate is a parameter in an optimization algorithm that determines the step size at each iteration.",
        image: "",
        content: [
          { id: 1,
            title: "",
            value: "学习率是梯度下降等优化算法中的一个关键参数，也称为超参数。它决定了每次迭代中模型参数更新的速度，对模型训练的成效有着至关重要的影响，因为它直接影响损失函数的收敛速度和效果。"
          },
          { id: 2,
            title: "",
            value: "合适的学习率能够让训练高效且稳定，从而提升模型性能。相反，过高或过低的学习率可能导致损失值震荡、发散，或收敛过慢甚至停滞等问题。因此，选择合适的学习率对于模型的训练效果至关重要。"
          },
        ]
      },
      { id: 1,
        name: ":: 学习率曲线",
        value: "这张图展示了学习率与损失函数之间的关系, 以下是总结：",
        image: "src/assets/chapter_three/lr.jpeg",
        content: [
          { id: 1,
            title: "平坦区域:",
            value: "在极低的学习率下，损失函数几乎没有变化，曲线开头的平坦区域展示了这一点。低学习率意味着模型的调整幅度很小，可能导致收敛缓慢或停滞，难以有效改善模型性能。"
          },
          { id: 2,
            title: "下降区域:",
            value: "随着学习率的增加，损失函数开始下降，曲线的下降部分反映了这一过程。此时，学习率有效地引导模型朝着最小损失值前进，模型更新幅度更大，性能随之提升。"
          },
          { id: 3,
            title: "最佳点:",
            value: "在曲线的某个点（用星标标出），此时的学习率使损失函数达到最小值，表示最佳学习率。在这一点上，收敛速度与稳定性之间达到了理想的平衡。"
          },
          { id: 4,
            title: "指数增长区域:",
            value: "超过最佳点后，随着学习率的继续增大，训练损失急剧上升。这是曲线的指数增长区域，过高的学习率导致模型超越最小值，可能引发训练过程中的发散和不稳定现象。"
          },
        ]
      },
      { id: 2,
        name: ":: 学习率过低 (lr=0.05)",
        value: "",
        image: "src/assets/chapter_three/lr_low.jpeg",
        content: [
          { id: 1,
            title: "",
            value: "当学习率过低时，每次梯度下降的迭代步伐非常小，尤其是在训练后期。尽管模型最终可能会收敛到最小值，但整个过程非常缓慢，导致训练时间延长，需要更多迭代才能实现最小损失。严重时，模型甚至可能停止优化，即使增加迭代次数也无济于事。"
          },
        ]
      },
      { id: 3,
        name: ":: 学习率过高 (lr=0.8)",
        value: "",
        image: "src/assets/chapter_three/lr_high.jpeg",
        content: [
          { id: 1,
            title: "",
            value: "当学习率过高时，模型每次迭代的步伐较大，可能导致模型在最小值附近左右震荡，甚至发散，导致训练不稳定，无法收敛，最终损失不降反升。"
          },
        ]
      },
      { id: 4,
        name: ":: 学习率刚刚好 (lr=0.2)",
        value: "",
        image: "src/assets/chapter_three/lr_right.jpeg",
        content: [
          { id: 1,
            title: "",
            value: "而当学习率设置合适时，模型可以平稳且持续地优化，不会超过最小损失值。这样的学习率在速度和准确性之间达到了完美平衡，模型能够快速收敛到最小值。可以形象地说，这就像找到了一条既安全又能快速下山的路径。"
          },
        ]
      },
    ]
  }
]