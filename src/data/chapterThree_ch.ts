export default [
  { id: 0, 
    name: "Linear Regression",
    sections: [
      { id: 0,
        name: "1. Linear Regression", 
        value: "Linear regression is a statistic method that models a straight-line relationship between 2 types of variables.",
        image: "",
        content: [
          { id: 1, 
            title: "直线关系",
            value: "线性回归是一种统计方法, 它模型化了变量之间的直线关系。它旨在拟合一条最佳代表数据的直线, 最小化观察值和预测值之间的差异。这种方法广泛用于做出预测和理解各种领域的趋势。"
          },
          { id: 2, 
            title: "两种类型的变量",
            value: "在线性回归的最简单形式中, 涉及两种类型的变量：自变量和因变量。自变量用于预测因变量的结果。这种方法也可以扩展为包括多个自变量, 从而增强模型的预测能力。"
          },
          { id: 3, 
            title: "机器学习中的线性回归",
            value: "在线性回归是机器学习中的基础算法, 同时也是更复杂模型的入门工具。它通常是最先教授的技术之一, 因为它简单易懂, 易于解释, 是理解预测建模基础的重要工具。在机器学习中, 线性回归用于基于输入数据预测连续的结果, 例如价格、温度或得分。"
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
            value: "这里是一个示例, 用于可视化工作时间和薪水之间的简单线性关系。蓝色的直线显示薪水与工作时间成正比, 每小时的薪水增加率为 $10。"
          },
          { id: 2, 
            title: "",
            value: "这条直线说明每增加一个工作小时, 薪水会增加 $10, 展示了一个简单的线性回归模型。该模型有效地捕捉并预测了这两个变量之间的线性关系。"
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
            value: "正态分布是一种用于描述实值随机变量的连续概率分布。简单来说, 它是一种描述数据自然分布方式的方法。当数据遵循正态分布时, 大多数值会集中在一个中心点, 即均值附近, 随着离中心点越来越远, 值的数量会逐渐减少。这形成了经典的“钟形曲线”, 该曲线在两侧对称。"
          },
          { id: 2, 
            title: "为什么我们要了解正态分布？",
            value: "正态分布在机器学习和深度学习中具有基础性作用。当我们构建第一个模型时, 我们使用正态分布来初始化权重和偏差, 为什么这么做？因为在深度学习中, 神经网络中的权重和偏差通常使用从正态分布中抽取的值来初始化。这有助于确保模型以平衡的方式开始训练, 防止出现梯度消失或梯度爆炸的问题。更多的细节我们会在梯度下降中进一步讲解。"
          },
          { id: 3, 
            title: "正态分布在机器学习中的应用",
            value: "在机器学习中, 线性回归是基础算法之一, 同时也是通向更复杂模型的门户。它通常是首先教授的技术之一, 因为它简单且易于解释, 使其成为理解预测建模基础的关键工具。在机器学习中, 线性回归用于根据输入数据预测连续的结果, 如价格、温度或分数。"
          },
        ]
      },
      { id: 1,
        name: "2.1 概率密度函数 (PDF)", 
        value: "让我们深入了解概率密度函数, 即 PDF, 它在谈论标准正态分布时非常关键。不需要去记住复杂的公式; 关键在于理解函数的形状和影响它的关键参数。这些参数决定了钟形曲线的宽度, 从而清楚地展示了数据的分布情况。",
        image: "src/assets/chapter_three/ndistrib_pdf.jpeg",
        content: [
          { id: 1, 
            title: "钟形曲线:",
            value: "想象一个钟。这就是我们谈论的形状。"
          },
          { id: 2, 
            title: "均值 (µ = 0):",
            value: "这是钟的顶点或最高点。在标准正态分布中, µ 等于 0。"
          },
          { id: 3, 
            title: "标准差 (σ = 1):",
            value: "这决定了钟的宽度。在标准正态分布中, σ 等于 1。"
          },
          { id: 4, 
            title: "x轴",
            value: "代表随机变量的可能值。"
          },
          { id: 5, 
            title: "y轴:",
            value: "显示概率密度, 或每个 x轴 上值的可能性。"
          },
        ]
      },
      { id: 2,
        name: "2.2 累积分布函数 (CDF)", 
        value: "累积分布函数 (CDF) 可以通过曲线下方特定点( 我们称之为 x) 左侧的阴影区域来可视化。它表示随机变量 X 取值小于或等于 x 的概率。",
        image: "src/assets/chapter_three/ndistrib_cdf.jpeg",
        content: [
          { id: 1, 
            title: "S形曲线:",
            value: "可以把它想象成一个 S 形曲线, 但仍然与钟形曲线相关。"
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
            value: "损失函数在评估模型预测与实际数据的匹配程度中至关重要。它衡量模型预测值与真实值之间的差异, 目标是将这种差异最小化。简单来说, 损失越低, 说明模型的预测越准确。"
          },
          { id: 2,
            title: "工作原理",
            value: "损失函数计算模型预测值与数据集中实际值之间的差异。这种差异被汇总为一个标量值, 表示“损失”或误差。"
          },
          { id: 3,
            title: "为什么它很重要",
            value: "损失函数在优化过程中发挥着关键作用。在训练过程中, 模型利用损失值来微调其内部参数( 如权重和偏差) , 以减少损失, 从而随着时间的推移提供更好的预测。"
          },
          { id: 4,
            title: "损失值越低越好",
            value: "训练中的关键目标是最小化损失函数。较低的损失表示模型的预测值更接近实际值, 说明模型正在有效学习。"
          }
        ]
      },
      { id: 1,
        name: "3.1 均方误差 (MSE)",
        value: "均方误差计算预测值与实际值之间差异的平方的平均值。由于较大的误差被平方, 因此MSE对异常值更为敏感, 使其显得更为重要。",
        image: "",
        content: [
          { id: 1,
            title: "",
            value: "It calculates the average of the squared differences between predicted and actual values. Since larger errors are squared, MSE is more sensitive to outliers, making them more significant."
          },
        ]
      },
      { id: 2,
        name: ":: 直线模型中的MSE( 第1部分) ",
        value: "",
        image: "src/assets/chapter_three/mse_one.jpeg",
        content: [
          { id: 1,
            title: "图形解释:",
            value: "该图显示了两条线：当前线( 红色) 代表模型的预测值, 目标线( 蓝色) 代表我们希望模型预测的真实值。不同x值的数据点被绘制为图上的点——蓝色点对应目标值, 而红色点显示了当前模型的预测值。"
          },
          { id: 2,
            title: "数学表示:",
            value: "当前线由方程  y = 0.3x + b  表示, 其中b是偏差项。目标线的方程为  Y = 0.6x + 0.1 。预测值与实际目标值之间的差异通过垂直箭头表示, 展示了两者之间的间隔。"
          },
        ]
      },
      { id: 3,
        name: ":: 直线模型中的MSE( 第2部分) ",
        value: "",
        image: "src/assets/chapter_three/mse_two.jpeg",
        content: [
          { id: 1,
            title: "MSE计算:",
            value: "均方误差( MSE) 通过平均预测值( 来自红线) 与实际目标值( 来自蓝线) 之间的平方差来计算。这个公式衡量了预测值与实际值之间的偏离程度。"
          },
          { id: 2, 
            title: "MSE图:",
            value: "右侧的图绘制了MSE作为偏差项b的函数。曲线显示了当b变化时MSE如何变化, 最低点表示最小化误差的b的最佳值。"
          },
          { id: 3, 
            title: "目标:",
            value: "训练模型的目标是调整偏差b( 从考虑一个变量开始) 以最小化MSE。这可以通过找到MSE曲线上的最低点来直观地表示。"
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
            value: "The Loss Curve is a graphic tool that tracks a model’s error or losses over the training period. It helps us easily understand how well a model is performing during both training and evaluation."
          },
          { id: 2,
            title: "",
            value: "损失曲线是一个图形工具, 用于跟踪模型在训练期间的错误或损失。它帮助我们轻松理解模型在训练和评估过程中表现如何。损失曲线直观地表示了损失在训练周期( epochs, 即训练迭代) 中的变化。通常, 你会看到同一图表上有两条曲线：训练损失曲线显示训练数据上的损失如何随epoch减少, 而测试损失曲线( 也称为验证损失曲线) 则展示了测试数据上的损失在相同时间段内的变化。"
          },
        ]
      },
      { id: 5,
        name: ":: 三种损失曲线",
        value: "",
        image: "src/assets/chapter_three/losscurves.jpg",
        content: [
          { id: 1,
            title: "欠拟合( Underfitting) :",
            value: "在第一个图表中, 标记为“欠拟合”, 训练损失( 蓝线) 和测试损失( 红线) 都很高且下降缓慢。两者之间的差距很小, 但都可以更低。这表明模型难以学习数据中的潜在模式。"
          },
          { id: 2,
            title: "过拟合( Overfitting) :",
            value: "第二个图表标记为“过拟合”, 显示了训练损失( 蓝线) 迅速下降且远低于测试损失( 红线) , 测试损失在初始下降后开始上升。这意味着模型在训练数据上表现非常好, 但在测试数据上表现较差, 说明模型对训练集过于专门化, 不能很好地泛化。"
          },
          { id: 3,
            title: "恰到好处( Just Right) :",
            value: "第三个图表标记为“恰到好处”, 训练损失( 蓝线) 和测试损失( 红线) 都平稳下降并趋于相似的低值。这表明模型调整得很好, 从训练数据中有效学习, 同时也能够良好地泛化到测试数据。"
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
            value: "想象你在一座山上, 目标是到达最低点, 这就是全局最小值。损失函数代表了地形, 而梯度则是斜坡, 指引你应该朝哪个方向迈步下山。梯度是损失函数相对于模型参数的导数, 表示损失函数变化的方向和速度。"
          },
        ]
      },
      { id: 1,
        name: "4.1 梯度下降模拟( 1个参数) ",
        value: "",
        image: "src/assets/chapter_three/gd_one.jpeg",
        content: [
          { id: 0,
            title: "",
            value: "该图形直观地展示了梯度下降( Gradient Descent) 如何通过迭代调整模型参数, 逐步减小模型预测值( 红点) 与实际值( 蓝点) 之间的误差。每一步中, 模型的预测线从 y = 0.3x 向接近目标线 Y = 0.6x + 0.1 移动, 从而有效降低总体误差。模型线逐步与目标线对齐, 展示了梯度下降在提升模型性能方面的有效性。"
          },
          { id: 1,
            title: "初始线( 红色) :",
            value: "红点表示模型在参数 b = 0 时做出的初始预测。红线代表模型在进行梯度下降迭代之前的起点。"
          },
          { id: 2,
            title: "目标线( 蓝色) :",
            value: "蓝点表示模型应预测的理想目标值。穿过这些点的蓝线是模型通过梯度下降希望逼近的最优线。"
          },
          { id: 3,
            title: "梯度下降迭代:",
            value: "红线与蓝线之间的灰线展示了每次梯度下降迭代后, 模型预测线的变化。每次更新都会减少模型预测值与目标值之间的差异, 逐步将预测线移向目标线。此过程演示了模型朝着更准确预测值的收敛过程。"
          },
        ]
      },
      { id: 2,
        name: ":: 梯度下降模拟表格",
        value: "",
        image: "src/assets/chapter_three/gd_one_table.jpeg",
        content: [
          { id: 1,
            title: "",
            value: "在这个表格中, 我们展示了参数b在若干次迭代中的变化, 以及对应的均方误差( MSE) 和MSE的导数。算法从b的初始猜测值( 为了演示方便设置为0) 开始, 并根据MSE在该点的梯度( 斜率) 迭代更新b。更新规则由以下公式给出：b(n+1) = b(n) - lr * MSE’(b), 其中lr是学习率, 我们将其设置为0.2。"
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
            value: "左边的图显示了MSE损失函数随着参数b而变化的曲线。该曲线表示当我们调整b时, 误差如何变化。曲线上的红点展示了梯度下降的进程, 从初始值b = 0开始, 每一步更新后b逐渐向更低的MSE值移动。与曲线相切的线表示每个点的梯度, 指导更新的方向和幅度。"
          },
          { id: 2,
            title: "MSE的导数",
            value: "右边的图表示MSE关于b的导数。红点表示每次迭代时梯度的数值。随着算法接近最小值, 梯度逐渐减小。当b接近最小化MSE的值时, 梯度趋近于零。当梯度为零时, 算法已达到b的最优值。"
          },
        ]
      },
      { id: 4,
        name: "4.2 梯度下降 (2个参数)",
        value: "这张图可视化了两个参数的梯度下降过程。这张3D曲面图很直观地展示了在两个参数场景中的梯度下降过程。",
        image: "src/assets/gradient_descent.jpeg",
        content: [
          { id: 1,
            title: "损失函数曲面图:",
            value: "图中展示了两个参数的损失函数曲线图, 曲面的高度表示不同 θ1 和 θ2 取值下损失函数的数值。梯度下降的目标是沿着这个曲面找到损失函数的最小点, 也就是曲面上的最低点 (全局最小值) 。"
          },
          { id: 2,
            title: "梯度下降过程:",
            value: "曲面上的箭头指示了梯度下降算法在调整 θ1 和 θ2 时的路径, 随着每次迭代逐步接近最小损失值。这些箭头展示了算法是如何沿着曲面的斜率 (梯度) 向低处移动的。"
          },
          { id: 3,
            title: "损失函数的等高线:",
            value: "图中还反映了损失函数的等高线, 深色区域表示较高的损失值, 浅色区域表示较低的损失值。梯度箭头从深色区域向浅色区域移动, 清楚地表示了损失值的优化过程。"
          },
          { id: 4,
            title: "坐标轴:",
            value: "x轴和y轴表示参数θ1和θ2, z轴为损失函数。"
          },
        ]
      },
      { id: 5,
        name: ":: 梯度下降计算表格",
        value: "",
        image: "src/assets/chapter_three/gd_two_table.jpeg",
        content: [
          { id: 1,
            title: "",
            value: "在这个表格中, 我们计算了两个参数 (偏置 b 和权重 w ), 以及对应的均方误差 (MSE) 和其导数, 在若干次迭代中的数值变化。这里我们初始化 b = 0, w = 0.3 。"
          },
          { id: 2,
            title: "",
            value: "接着, 我们根据 MSE 的梯度来更新 w 和 b。更新规则与单一参数损失函数相似。这里的需要注意的是: 当我们计算一个参数的斜率时, 将另一个参数视为常量处理。"
          },
        ]
      },
      { id: 6,
        name: ":: 梯度下降过程可视化",
        value: "",
        image: "src/assets/chapter_three/gd_two.jpeg",
        content: [
          { id: 1,
            title: "",
            value: "这张图展示了, 梯度下降如何通过迭代调整线性模型的两个参数：权重 (w) 和偏置 (b), 从而优化模型。"
          },
          { id: 2,
            title: "",
            value: "之前我们只考虑了一个参数 (偏置b) 的损失函数, 并将权重固定在 w = 0.3。这帮助我们观察了梯度下降是如何调整偏置以减少误差的。现在, 我们将这两个参数一起考虑, 参与模型优化过程。"
          },
          { id: 3,
            title: "",
            value: "当这两个参数都在损失函数中的时候, 梯度下降算法在每次迭代中会同时更新它们, 逐步减少模型的误差, 这和一个参数的损失函数迭代过程相似。唯一的区别是, 这次我们需要同时计算 w 和 b 的梯度, 并相应地调整模型。"
          },
          { id: 4,
            title: "",
            value: "该图很形象地可视化了模型在每次迭代中的变化, 每一次迭代, 参数更新之后当前模型得以调整, 从而靠近目标些许。与之前的单一变量类似, 这里我们又加入了一个变量, 需要更多的计算, 而原理则基本保持不变。"
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
            value: "随机梯度下降(SGD) 是一种机器学习中的优化算法。与常规的梯度下降不同, SGD 在每次迭代时只使用一个样本或一小部分随机样本来更新模型参数, 而不是像梯度下降一样使用整个数据集。这样可以使损失函数更加快速地收敛, 但也会引入更大的波动的不稳定性。"
          },
          { id: 2,
            title: "为什么使用 SGD",
            value: "SGD 比常规梯度下降更快且节省内存, 尤其适用于大规模的数据集。因为 SGD 仅使用一个或小部分数据进行更新, SGD 能够更快且更频繁地调整参数。然而, 由于它使用的是较小的数据集, 路径可能会比较崎岖, 并且收敛过程不如梯度下降平滑稳定。"
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
            value: "梯度下降( GD) 在处理大型数据集时表现不佳, 因为它每次更新都要处理整个数据集来计算损失函数。随机梯度下降( SGD) 则不需要, 它每次只需使用一个样本或一小部分样本来更新参数, 所以对于大数据集也可以快速计算得到损失值。"
          },
          { id: 2,
            title: "大量参数:",
            value: "GD 能很好地处理大量参数, 但由于需要同时计算所有参数的梯度, 速度可能较慢。SGD 由于更新频率更高, 在处理众多参数时更具扩展性。"
          },
          { id: 3,
            title: "参数更新:",
            value: "GD 在处理完整个数据集后才更新参数, 导致更新稳定且准确。SGD 每次根据一个样本或小部分样本更新参数, 虽然速度更快, 但也带来了一定的噪声。"
          },
          { id: 4,
            title: "RAM 使用:",
            value: "GD 每次需处理一个完整的数据集, 因此需要更多的计算内存。SGD 只处理少量数据, 因此对内存的压力小很多。"
          },
          { id: 5,
            title: "收敛速度:",
            value: "GD 收敛较慢, 但过程更稳定, 能更精确地接近最小值。SGD 收敛速度更快, 但由于更新过程中有波动, 可能在最小值附近发生振荡。"
          },
          { id: 6,
            title: "准确性:",
            value: "GD 到最后通常可以有高准确性, 因为更新过程没有多少噪声, 能更好地反映数据的总体趋势。SGD 由于更新过程常伴随噪声, 准确性略低, 但可通过学习率衰减等技术配合改善。"
          },
        ]
      },
      { id: 2,
        name: ":: 随机梯度下降与梯度下降 (part.2)",
        value: "这幅图清晰地说明了梯度下降( GD) 和随机梯度下降( SGD) 在模型训练时的不同之处。",
        image: "src/assets/chapter_three/sgd_two.jpeg",
        content: [
          { id: 1,
            title: "梯度下降( GD) :",
            value: "图中可以看出直到中心最优点, 其步伐都很稳定, 但在处理大数据集时, 就会变得过于缓慢, 且极其极消耗计算成本。"
          },
          { id: 2,
            title: "随机梯度下降( SGD) :",
            value: "一个提供了快速且更节省内存的替代方案, 使用随机梯度下降我们可以快速找到最优点, 虽然步伐有些飘忽, 专业术语为噪声, 这让其通向最小值的路径有了些不稳定和不可预测性。这种噪声有助于跳出局部最小值 (可以参考梯度下降部分的二维曲面图), 这时我们可能需要用学习率递减等技术来稳定最终的收敛过程。"
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
            value: "学习率是梯度下降等优化算法中的一个重要参数, 我们也称其为超参数。它决定了在每次迭代中模型参数更新的快慢。学习率对模型训练有着极其重要的影响, 它决定了模型损失函数的收敛速度和收敛效果。"
          },
          { id: 2,
            title: "",
            value: "合适的学习率可以使训练高效且稳定, 从而提高模型性能。相反, 过高或过低的学习率可能导致诸如损失震荡、发散, 或者收敛过慢甚至停滞等问题。因此, 设置合适的学习率对于模型的训练效果至关重要。"
          },
        ]
      },
      { id: 1,
        name: ":: 学习率曲线",
        value: "该图展示了学习率与损失函数之间的关系, 以下是总结：",
        image: "src/assets/chapter_three/lr.jpeg",
        content: [
          { id: 1,
            title: "平坦区域:",
            value: "在极低的学习率下, 损失函数几乎保持不变, 曲线开头的平坦区域显示了这一点。低学习率意味着模型的调整幅度很小, 可能导致收敛延迟和停滞不前而难以得到有效的改善。"
          },
          { id: 2,
            title: "下降区域:",
            value: "随着学习率的增加, 损失函数开始下降, 这在曲线的下降部分表现出来。这一阶段的学习率有效地引导模型朝着最小损失方向前进, 模型做出更大的更新, 从而提升其性能。"
          },
          { id: 3,
            title: "最佳点:",
            value: "在曲线上的某个点( 用星标表示) , 这个时候的学习率可以使损失函数达到最小值。这代表了最佳学习率, 此时函数的收敛速度和稳定性之间的平衡是最理想的。"
          },
          { id: 4,
            title: "指数增长区域:",
            value: "超过最佳点后, 随着学习率的继续增加, 训练损失急剧上升。这是曲线的指数增长区域, 学习率过高导致模型超越最小值, 可能导致训练过程中的发散和不稳定。"
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
            value: "当学习率过低时, 每次梯度下降的迭代步骤会变得非常小, 尤其是在训练的后期阶段。虽然模型最终可能会收敛到最小值, 但这个过程非常缓慢, 导致训练时间变长, 并且需要更多的迭代次数才能实现最小损失。严重的时候, 模型甚至会停止优化, 即使迭代再多的次数也无济于事。"
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
            value: "当学习率设置得过高时, 模型在每次迭代中迈出的步伐就比较大。这可能导致模型在最小值附近左右震荡发散, 造成模型训练不稳定, 无法收敛, 甚至导致损失增加而不是减少。"
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
            value: "当学习率合适时, 模型可以在不超过最小损失值的情况下平稳且持续地优化。这种模型优化的速度和平稳性使模型能够快速地收敛到最小损失值。此时设置的学习率在速度和准确性之间找到了完美的平衡, 形象得说就是我们找到了一条即安全又可以快速下山的路。"
          },
        ]
      },
    ]
  }
]