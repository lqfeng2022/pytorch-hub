export default [
  { id: 1, 
    name: "TENSORs ?",
    sections: [
      { id: 0,
        name: "什么是张量", 
        value: "A TENSOR is a multi-dimensional array with a single data type.",
        image: "",
        content: [
          { id: 1, 
            title: "",
            value: "张量是一个数学概念, 它将标量、向量和矩阵的概念扩展到更高维度。张量在物理、工程和机器学习等领域非常重要, 因为张量让我们能够以有序的方式处理复杂的多维数据。"
          },
        ]
      },
      { id: 1,
        name: ":: 标量, 向量, 矩阵, 张量", 
        value: "",
        image: "src/assets/chapter_one/tensors.jpeg",
        content: [
          { id: 1, 
            title: "标量",
            value: "标量只是一个的数字( 0维张量) , 为了方便理解, 可以将其看作一个屏幕上像素点。通常我们用小写字母表示标量。"
          },
          { id: 2, 
            title: "向量",
            value: "向量是一组标量, 一个列表( 1维张量) , 可以想象成一条直线 (水平方向连续的像素点)。我们用加粗的小写字母表示向量。"
          },
          { id: 3, 
            title: "矩阵",
            value: "矩阵是一组列表( 2维张量) , 可以将其视为一个正方形或矩形。在我们的例子中, 有3行, 每行有3列数字。通常我们用大写字母表示矩阵。"
          },
          { id: 4, 
            title: "张量",
            value: "张量将这个概念拓展到了三维, 四位以及更高的纬度。例如, 一个三维张量表示一组矩阵堆栈在一起。我们通常用加粗的大写字母来表示张量。"
          },
        ]
      },
    ]
  },
  { id: 2, 
    name: "Tensor Creating",
    sections: [
      { id: 0,
        name: "2. 创建张量", 
        value: "",
        image: "src/assets/chapter_one/create.jpeg",
        content: [
          { id: 1, 
            title: "",
            value: "我们可以创建一个充满随机数的张量, 或者设置一个限定范围内的张量。你也可以创建一个全部元素为 1 或 0 的张量, 还可以创建一个与现有张量形状相同的全部为 1 或 0 的新的张量。"
          },
        ]
      },
      { id: 1,
        name: "2.1 随机张量", 
        value: "",
        image: "src/assets/chapter_one/create_random.jpeg",
        content: [
          { id: 1, 
            title: "什么是随机张量？",
            value: "随机张量就是一个充满随机数的张量。这些随机数通常来自特定的概率分布, 比如正态分布——大多数数字集中在均值附近, 或均匀分布——在某个范围内的所有数字都有相同的被选中概率。"
          },
          { id: 2, 
            title: "为什么使用随机张量？",
            value: "随机张量在机器学习和深度学习中有许多用途, 例如初始化权重。当你设置神经网络时, 网络层的权重需要有一些初始值, 通常这些初始值是小的随机数。随机值的初始化有助于启动学习过程, 防止对称性问题, 从而让网络更有效地学习。"
          },
          { id: 3, 
            title: "如何创建随机张量？",
            value: "在 PyTorch 中, 你可以使用像 torch.rand() 这样的函数来创建均匀分布的随机张量, 使用 torch.randn() 来创建正态分布的随机张量, 或者根据需求使用其他函数。"
          },
        ]
      },
      { id: 2,
        name: "2.2 零张量 & 一张量", 
        value: "",
        image: "src/assets/chapter_one/create_zerosOnes.jpeg",
        content: [
          { id: 1, 
            title: "什么是零张量和一张量？",
            value: "零张量或一张量是每个元素都被初始化为零或一的张量。可以将它们想象成充满零或一的数组, 但它们可以是多维的, 也更复杂。"
          },
          { id: 2, 
            title: "为什么使用零张量和一张量？",
            value: "在神经网络中, 偏置项( 参数) 通常初始化为零, 因为这不会在训练过程中引入任何不对称或偏好。有时你可能需要将某些参数或张量初始化为一, 而不是零, 特别是在涉及乘法运算时。还有很多其他原因, 我们将在具体的场景中逐步介绍。"
          },
          { id: 3, 
            title: "如何创建零张量和一张量？",
            value: "在 PyTorch 中, 你可以使用 torch.zeros() 和 torch.ones() 来创建零张量和一张量。例如, torch.zeros(3, 3) 和 torch.ones(3, 3) 会分别生成两个 3x3 的矩阵, 分别填充为全零或全一。"
          },
        ]
      },
      { id: 3,
        name: "2.3 范围张量", 
        value: "",
        image: "src/assets/chapter_one/create_range.jpeg",
        content: [
          { id: 1, 
            title: "什么是范围张量？",
            value: "范围张量是一个包含在指定范围和步长内的数字序列的张量。它类似于 Python 中的 range() 函数, 但它生成的是张量而不是列表。"
          },
          { id: 2, 
            title: "为什么使用范围张量？",
            value: "范围张量通常用于在其他张量中进行切片或操作特定范围时创建索引值。"
          },
          { id: 3, 
            title: "如何创建范围张量？",
            value: "在 PyTorch 中, 你可以使用 torch.arange(start, end, step) 函数来创建范围张量。其中 start 表示序列的起始值( 包含) , end 表示序列的结束值( 不包含) , step 表示每个相邻值之间的差异。"
          },
        ]
      },
      { id: 4,
        name: "2.4 类似张量", 
        value: "",
        image: "src/assets/chapter_one/create_like.jpeg",
        content: [
          { id: 1, 
            title: "",
            value: "在 PyTorch 中, 当我们提到 ‘类似张量’ 时, 通常指的是像 torch.zeros_like() 和 torch.ones_like() 这样的函数。这些函数用于创建与给定张量具有相同形状和设备的新张量, 但这些张量的元素全部被初始化为零或一。"
          },
        ]
      },
    ]
  },
  { id: 3, 
    name: "Tensor Attributes",
    sections: [
      { id: 0,
        name: "3. 张量属性", 
        value: "在 PyTorch 中, 张量有几个重要的属性, 它们可以提供有关张量结构、数据类型以及存储位置( CPU 或 GPU) 的信息。你经常会使用的主要属性有 ‘shape’( 形状) 、‘ndim’( 维度) 、‘dtype’( 数据类型) 和 ‘device’( 设备) 。以下是每个属性的简要概述:",
        image: "src/assets/chapter_one/attributes.jpeg",
        content: [
          { id: 1, 
            title: "形状 (shape)",
            value: "‘形状’ 属性告诉你张量每个维度的大小。它是一个整数元组, 每个整数代表张量在某一特定维度上的大小。了解张量的形状非常重要, 因为它定义了你正在处理的数据的结构, 并告诉你每个维度包含多少元素。"
          },
          { id: 2, 
            title: "维度 (ndim)",
            value: "‘ndim’ 属性表示张量的维数( 也称为轴的数量) 。理解张量的维度对操作如重塑( reshaping) 和可视化数据结构非常关键。"
          },
          { id: 3, 
            title: "数据类型 (dtype)",
            value: "‘dtype’ 属性指示存储在张量中的元素的数据类型, 比如 ‘torch.float32’ 或 ‘torch.int64’。数据类型决定了张量中数值的精度和范围。选择合适的数据类型会影响计算的性能和内存使用。"
          },
          { id: 4, 
            title: "设备 (device)",
            value: "‘设备’ 属性告诉你张量的数据存储在何处：CPU 或 GPU。存储在 GPU 上的张量, 其设备属性会显示为类似 ‘cuda:0’ 的值。在使用 GPU 加速计算时, 了解张量存储的位置非常重要。为了避免错误, 你需要确保参与计算的所有张量都位于相同的设备上。"
          },
        ]
      },
      { id: 1,
        name: "3.1 形状 (Shape)", 
        value: "在 PyTorch 中使用 shape 属性有时会比较棘手, 特别是在处理张量的操作或执行涉及重塑和矩阵乘法的操作时。我们通过一个例子来帮助可视化并理解 shape 属性。",
        image: "src/assets/chapter_one/attributes_shape.jpeg",
        content: [
          { id: 1, 
            title: "",
            value: "在这个例子中, 我们使用 torch.rand(3, 4) 创建了一个张量 X。这个张量的 shape 属性是 torch.Size([3, 4])。第一个数字 (3) 表示 外部维度, 即行数。第二个数字 (4) 表示 内部维度, 即列数。因此, 该张量的结构就像一个 3x4 的网格, 每个单元格中包含一个随机数。"
          },
          { id: 2, 
            title: "",
            value: "从外部维度到内部维度的这种模式类似于从网格中的行到列的顺序。ndim 属性告诉我们张量的维度数。在这个例子中, X.ndim 是 2, 表示这是一个二维张量, 符合网格结构。默认情况下, 该张量的数据类型 (dtype) 是 torch.float32, 并且它存储在 CPU (device) 上。"
          },
        ]
      },
    ]
  },
  { id: 4, 
    name: "Tensor Operations",
    sections: [
      { id: 0,
        name: "4. 张量运算 (Tensor Operations)", 
        value: "",
        image: "src/assets/chapter_one/operats.jpeg",
        content: [
          { id: 1, 
            title: "",
            value: "在机器学习的世界中, 张量是表示复杂数据的基本构建块。张量操作让我们能够高效地处理这些数据, 从基本的算术运算到高级的变换。正如我们对标量、向量和矩阵执行操作一样, 我们也可以对张量应用类似的操作, 这使得它们在重塑数据、执行矩阵乘法等任务中至关重要。"
          },
        ]
      },
      { id: 1,
        name: "4.1 基本运算", 
        value: "",
        image: "src/assets/chapter_one/operats_addsub.jpeg",
        content: [
          { id: 1, 
            title: "",
            value: "你可以将一个数字加到或从一个张量中减去, 这将对张量的每个元素应用该操作。此外, 你还可以通过逐元素地执行操作来加减两个形状相同的张量。与加法和减法类似, 你可以将张量的每个元素乘以一个数字, 或与另一个形状相同的张量执行逐元素的乘法或除法。"
          },
        ]
      },
      { id: 2,
        name: "4.2 矩阵乘法", 
        value: "",
        image: "src/assets/chapter_one/operats_matmul.jpeg",
        content: [
          { id: 1, 
            title: "",
            value: "矩阵乘法是线性代数中的一个基本操作, 其中两个矩阵相乘以生成第三个矩阵。这个操作在各种领域中都至关重要, 包括计算机科学、工程、物理学, 特别是在机器学习中, 它在训练神经网络、数据转换和高效处理大型数据集等任务中发挥了关键作用。"
          },
          { id: 2, 
            title: "",
            value: "矩阵乘法, 也称为点积( 当应用于向量时) , 涉及将第一个矩阵的一行和第二个矩阵的一列对应元素的乘积求和。要使矩阵乘法有效, 第一个矩阵的列数必须等于第二个矩阵的行数。"
          },
        ]
      },
      { id: 3,
        name: ":: 如何运作", 
        value: "",
        image: "src/assets/chapter_one/operats_matmul_work.jpeg",
        content: [
          { id: 1, 
            title: "",
            value: "要乘以矩阵 A 和 B, 我们计算 A 的每一行与 B 的每一列的点积。得到的矩阵 C 的形状为 [2, 2]。矩阵 C 的每个元素是通过将 A 的一行与 B 的相应列点乘得到的：C 的第一行是 A 的第一行与 B 的第一列和第二列的点积, C 的第二行是 A 的第二行与 B 的第一列和第二列的点积。"
          },
        ]
      },
      { id: 4,
        name: ":: 两个规则", 
        value: "",
        image: "src/assets/chapter_one/operats_matmul_rules.jpeg",
        content: [
          { id: 1, 
            title: "",
            value: "要进行矩阵乘法, 第一个矩阵( A) 的列数必须等于第二个矩阵( B) 的行数。如果 A 是一个 2 x 3 的矩阵, B 是一个 3 x p 的矩阵, 则结果矩阵( C) 的形状将是 2 x p。"
          },
        ]
      },
      { id: 5,
        name: ":: 矩阵乘法学习网站", 
        value: "如果你想深入了解矩阵乘法, 我推荐查看以下两个网站。它们提供了有价值的见解, 可以增强你对矩阵乘法和点积的理解。",
        image: "src/assets/chapter_one/operats_matmul_webs.jpeg",
        content: [
          { id: 1, 
            title: "mathisfun.com",
            value: "Math is Fun 是一个很棒的网站, 用于以清晰和有趣的方式学习和可视化数学概念。如果你想了解矩阵乘法和点积, 这个网站提供了简单但清晰的解释、示例和互动工具, 使这些概念易于理解。"
          },
          { id: 2, 
            title: "matrixmultiplication.xyz",
            value: "这是一个专注于帮助用户互动理解和执行矩阵乘法的工具。它允许你查看计算过程的每一步, 使你更容易掌握矩阵乘法的工作原理。"
          },
        ]
      },
      { id: 6,
        name: ":: 点积", 
        value: "",
        image: "src/assets/chapter_one/operats_matmul_dot.jpeg",
        content: [
          { id: 1, 
            title: "",
            value: "点积是向量代数中的一个基本操作, 广泛应用于物理学、工程学、计算机科学和机器学习等领域。它将两个向量相乘以产生一个标量值, 提供了有关向量之间关系的见解。点积可以揭示向量的对齐程度, 指示它们是否指向相同或相反的方向, 还可以确定向量是否垂直, 即它们是否在直角处相遇。"
          },
        ]
      },
      { id: 7,
        name: ":: Transformer 中的点积", 
        value: "点积是 Transformer 架构中的一个基本操作, 使模型能够高效、有效地权衡输入的不同部分的重要性, 这也是它在翻译、摘要等任务中变得如此强大的原因。",
        image: "src/assets/chapter_one/operats_matmul_dot_mha.jpeg",
        content: [
          { id: 1, 
            title: "相似性测量",
            value: "在机器学习中, 点积是测量两个向量之间相似性的关键操作。例如, 在推荐系统中, 点积可以量化用户的偏好与项目特征的匹配程度。这个概念是余弦相似度等技术的基础, 余弦相似度通过对点积进行归一化来考虑向量的大小。"
          },
          { id: 2, 
            title: "优化",
            value: "在优化和深度学习中, 点积在损失函数和梯度计算中的各种操作中起着核心作用。它在反向传播中扮演着重要角色, 在这个过程中, 计算梯度以最小化损失函数并有效更新模型参数。"
          },
          { id: 3, 
            title: "特征提取和表示",
            value: "点积也用于特征提取和表示。例如, 在神经网络中, 点积用于层内将输入特征与权重相结合, 产生激活值, 然后通过非线性函数处理。这一操作是网络学习数据中复杂模式的基础。"
          },
        ]
      },
      { id: 8,
        name: "4.3 聚合运算", 
        value: "聚合运算是指沿一个或多个维度减少张量的过程, 以生成一个较低秩的张量。聚合在深度学习中常用于总结或压缩张量中包含的信息, 使其更易于进一步处理。",
        image: "src/assets/chapter_one/operats_aggregate.jpeg",
        content: [
          { id: 1, 
            title: "max()/min()",
            value: "找到张量中所有元素或沿特定轴的最大值/最小值。"
          },
          { id: 2, 
            title: "sum()/mean()",
            value: "计算张量中所有元素或沿特定轴的总和/平均值。"
          },
          { id: 3, 
            title: "argmax()/argmin()",
            value: "返回张量沿特定轴的最大值/最小值的索引。"
          },
        ]
      },
    ]
  },
  { id: 5, 
    name: "Tensor Manipulation",
    sections: [
      { id: 0,
        name: "5. 张量操作", 
        value: "",
        image: "src/assets/chapter_one/manipul.jpeg",
        content: [
          { id: 1, 
            title: "",
            value: "我们讨论了各种张量操作, 如加法、减法、除法、逐元素乘法和矩阵乘法, 这些在深度学习中是基础操作。这些操作是一个更广泛概念的组成部分, 称为张量操作。"
          },
          { id: 2, 
            title: "",
            value: "张量操作包括一系列技术, 用于聚合、重塑、压缩、扩展和排列张量, 改变它们的数据类型, 并执行诸如连接和堆叠等操作。在深度学习中, 掌握张量操作对于数据准备、模型构建和执行计算等任务至关重要。"
          },
        ]
      },
      { id: 1,
        name: "5.1 重塑张量", 
        value: "",
        image: "src/assets/chapter_one/manipul_shape.jpeg",
        content: [
          { id: 1, 
            title: "压缩或扩展张量",
            value: "在 PyTorch 中, squeeze() 方法用于去除张量中大小为 1 的维度, 有效地减少不必要的维度。与之相对, unsqueeze() 在指定位置添加一个大小为 1 的维度, 允许你根据需要扩展张量的形状, 以便进行批处理或通道扩展等操作。"
          },
          { id: 2, 
            title: "转置张量",
            value: "当我们转置一个张量时, 有两种选择 - transpose() 和 T。PyTorch 中的 transpose() 方法交换张量的两个指定维度, 从而灵活地重新排列数据。T 属性是转置 2D 张量最后两个维度的简写, 提供了一种快速转置矩阵的方法, 而无需指定维度。"
          },
          { id: 3, 
            title: "置换张量",
            value: "在 PyTorch 中置换张量意味着根据指定的顺序重新排列其维度。当你需要更改维度的顺序以适应重塑、对齐数据与模型要求或简单地重新排序数据维度时, 这个操作特别有用。"
          },
          { id: 4, 
            title: "重塑或查看张量",
            value: "PyTorch 中的 reshape() 和 view() 方法都可以在不改变数据的情况下更改张量的形状。reshape() 方法可以处理不连续的张量, 必要时会返回一个新张量, 而 view() 方法要求张量在内存中是连续的, 虽然更快但灵活性较差。"
          },
        ]
      },
      { id: 2,
        name: ":: 转置张量", 
        value: "在 PyTorch 中, 转置张量是指交换其两个维度。这个操作在数学计算中很常见, 特别是在线性代数中, 当需要交换矩阵的行和列时。但这不仅限于矩阵——你可以对任何张量进行转置, 无论它有多少维度。",
        image: "src/assets/chapter_one/manipul_shape_trans.jpeg",
        content: [
          { id: 1, 
            title: "",
            value: "转置不仅适用于 2D 张量——你可以交换高维张量中的任何两个维度。就像在 2D 张量中交换行和列一样, 你也可以使用转置来重新排列高维张量中的任意一对维度。"
          },
          { id: 2, 
            title: "",
            value: "在 PyTorch 中, 有两种方法可以转置张量。第一种方法是使用 torch.transpose(dim0, dim1) 函数, 它允许你交换张量的任何两个维度, 非常灵活, 适用于任何大小的张量。第二种方法是简写的 tensor.T, 它专门用于 2D 张量, 简单地交换行和列。"
          },
          { id: 3, 
            title: "",
            value: "那么, 什么时候使用转置操作呢？一个常见的场景是矩阵运算, 在矩阵乘法等操作中, 转置是关键的, 因为它可以正确对齐维度。"
          },
        ]
      },
      { id: 3,
        name: ":: 转置张量", 
        value: "",
        image: "src/assets/chapter_one/manipul_shape_permute.jpeg",
        content: [
          { id: 1, 
            title: "",
            value: "permute() 方法用于根据维度的索引位置重新排序张量的维度。这在需要为了特定操作而调整张量形状时至关重要, 以确保数据结构与后续计算的要求对齐。"
          },
        ]
      },
      { id: 4,
        name: ":: 重塑张量", 
        value: "",
        image: "src/assets/chapter_one/manipul_shape_reshape.jpeg",
        content: [
          { id: 1, 
            title: "",
            value: "这张图片展示了 reshape() 和 view() 如何重新排列元素以适应新的形状( 对原始维度没有限制) , 提供了在不改变基础数据本身的情况下, 如何灵活构造数据的方式。"
          },
          { id: 2, 
            title: "[3, 4, 5] --> [5, 4, 3]",
            value: "重塑为 [5, 4, 3] 保持了相同的维度数量( 3D) , 但改变了这些维度上元素的分组方式。"
          },
          { id: 3, 
            title: "[3, 4, 5] --> [3, 12]",
            value: "当重塑为 [3, 12] 时, 张量被减少为 2 维形状。这将最后两个维度( 4 和 5) 展平为一个 12 维的单一维度, 结果是一个低维度的张量。"
          },
          { id: 4, 
            title: "[3, 4, 5] --> [2, 2, 3, 5]",
            value: "当重塑为 [2, 2, 3, 5] 时, 张量被扩展为 4 维形状。这里, 原始张量被拆分成较小的块, 将维度从 3 增加到 4。"
          },
          { id: 5, 
            title: "reshape() vs. view()",
            value: "在 PyTorch 中, reshape() 可以改变张量的形状, 并在必要时创建数据的新副本, 因此更具灵活性。相反, view() 仅在数据的内存布局允许时更改形状而不进行复制, 因此更快但灵活性较差。"
          },
        ]
      },
      { id: 5,
        name: "5.2 更改数据类型", 
        value: "PyTorch 张量可以轻松与 NumPy 数组交互, 使得在两者之间转换变得简单。这种无缝的互操作性在你希望利用 PyTorch 的功能和 NumPy 的优势时非常有用。以下是如何实现这种交互的方式：",
        image: "src/assets/chapter_one/manipul_dtype.jpeg",
        content: [
          { id: 0, 
            title: "",
            value: "我们可以使用 type() 方法轻松地更改张量的数据类型。很简单, 对吧？然而, 当你将来自其他库的数据( 如 NumPy 数组) 转换为 PyTorch 张量时, 这种转换也可能自动发生。张量的数据类型可能不是你所期望的。虽然 PyTorch 张量默认为 float32, 但 NumPy 数组可能引入不同的数据类型( 如 float64) 。因此, 当同时处理 PyTorch 张量和 NumPy 数组时, 了解这些转换中的数据类型处理方式非常重要。"
          },
          { id: 1, 
            title: "使用 type() 更改数据类型",
            value: "更改张量的数据类型是 PyTorch 中的一个常见操作。你可能需要更改张量的数据类型以与某些操作兼容或优化内存使用。在这里, 我们使用 type() 更改张量的目标数据类型。更改张量的数据类型可以更好地控制计算效率和机器学习模型的准确性。"
          },
          { id: 2, 
            title: "NumPy 数组 -> PyTorch 张量",
            value: "要将 NumPy 数组转换为 PyTorch 张量, 可以使用 torch.from_numpy() 函数。与 numpy() 方法一样, 这种转换创建了一个与 NumPy 数组共享相同数据的张量, 确保 NumPy 数组中的任何更改也会反映在张量中。"
          },
          { id: 3, 
            title: "PyTorch 张量 -> NumPy 数组",
            value: "你可以使用 numpy() 方法将 PyTorch 张量转换为 NumPy 数组。此方法创建了张量数据的视图, 这意味着 NumPy 数组和 PyTorch 张量共享相同的基础数据。因此, 对一个的任何更改都会影响另一个。"
          },
        ]
      },
      // Move tensor run on GPUs here,
      // cus that's all about changing the device (cpu/gpu)
      { id: 6,
        name: "5.3 连接和堆叠张量", 
        value: "",
        image: "src/assets/chapter_one/manipul_catstack.jpeg",
        content: [
          { id: 1, 
            title: "连接张量",
            value: "连接多个张量沿现有维度, 将它们合并成一个单一的张量, 而不会添加新的轴。这对于沿共享轴连接数据非常理想, 例如合并数据批次。"
          },
          { id: 2, 
            title: "堆叠张量",
            value: "从这张图片中, 我们可以看到这个操作将 A 和 B 在第 1 维( 列) 上水平堆叠。结果张量 C 的形状为 [4, 10], 这意味着它有 4 行和 10 列。你可以把它想象成将 B 直接放在 A 旁边, 从而使列数翻倍。"
          },
        ]
      },
      { id: 7,
        name: ":: 堆叠张量", 
        value: "",
        image: "src/assets/chapter_one/manipul_stack.jpeg",
        content: [
          { id: 1, 
            title: "",
            value: "在这张图片中, 你可以看到两个张量 A 和 B 是如何合并成一个新的张量 C 的。原始张量的形状都是 [4, 5], 它们以一种添加额外维度的方式结合在一起。结果张量 C 的形状现在是 [2, 4, 5], 表示它包含两个层, 每个层的原始形状为 4 x 5。"
          },
          { id: 2, 
            title: "",
            value: "这个过程实际上是将 A 和 B 叠加在一起, 创建了一个新的结构, 其中第一个维度表示层的数量。每一层对应于一个原始张量, 整齐地保留了它们的内部排列, 同时将它们组合在一起。"
          },
        ]
      },
      { id: 8,
        name: ":: 垂直连接张量", 
        value: "",
        image: "src/assets/chapter_one/manipul_shape_cat_vstack.jpeg",
        content: [
          { id: 1, 
            title: "",
            value: "在这张图片中, 你可以观察到两个张量 A 和 B 是如何通过沿第一个维度垂直堆叠的方式组合在一起的。结果张量 C 的形状为 [8, 5], 这表明该操作通过将 B 直接放置在 A 下面, 实际将行数加倍。"
          },
          { id: 2, 
            title: "",
            value: "以类似的方式, 另一种方法也将 A 和 B 垂直堆叠, 产生相同的结果, 形状为 [8, 5]。这种方法提供了一种更简洁的方式来实现相同的结果, 顺畅地将两个张量连接成一个结构。"
          },
        ]
      },
      { id: 9,
        name: ":: 水平连接张量", 
        value: "",
        image: "src/assets/chapter_one/manipul_shape_cat_hstack.jpeg",
        content: [
          { id: 1, 
            title: "",
            value: "在这张图片中, 你可以看到张量 A 和 B 是如何通过沿第二个维度并排对齐的方式组合在一起的。结果张量 C 的形状为 [4, 10], 其中列数已加倍, 而行数保持不变。"
          },
          { id: 2, 
            title: "",
            value: "同样, 另一种方法也实现了相同的水平堆叠, 得到一个形状为 [4, 10] 的张量。这种方法提供了一种更简单的方式来完成相同的结果, 有效地将两个张量连接成一个更宽的结构。"
          },
        ]
      },
    ]
  },
  { id: 6, 
    name: "Tensor Indexing",
    sections: [
      { id: 0,
        name: "6. 张量索引", 
        value: "Tensor indexing in PyTorch is all about accessing specific elements, rows, columns, or subarrays within a tensor.",
        image: "",
        content: [
          { id: 1, 
            title: "",
            value: "在 PyTorch 中, 张量索引是访问张量中特定元素、行、列或子数组的操作。它类似于在 Python 中索引列表或数组, 但由于 PyTorch 支持多维张量, 因此功能更强大、更灵活。无论你处理的是简单的 2D 矩阵还是更复杂的张量, 索引都可以让你深入到所需的精确数据。"
          },
        ]
      },
      { id: 1,
        name: "6.1 基本索引", 
        value: "",
        image: "src/assets/chapter_one/indexing_one.jpeg",
        content: [
          { id: 1, 
            title: "单个元素",
            value: "你可以使用索引访问张量的单个元素。就像 Python 中的列表一样, 你需要指定每个维度中的位置。"
          },
          { id: 2, 
            title: "行或列",
            value: "如果你想访问整个行或列, 可以通过为一个维度指定索引并使用“:”来访问另一个维度。这是一种快速获取特定轴上所有元素的方式。"
          },
        ]
      },
      { id: 2,
        name: "6.2 分割和布尔索引", 
        value: "",
        image: "src/assets/chapter_one/indexing_two.jpeg",
        content: [
          { id: 1, 
            title: "分割索引",
            value: "你可以使用切片从张量中提取一系列元素, 这类似于在 Python 中切片列表。你甚至可以指定步长值, 以跳过范围内的元素。"
          },
          { id: 2, 
            title: "布尔索引",
            value: "你还可以使用条件来索引张量, 这将返回所有符合该条件的元素。这对于根据特定标准过滤数据非常有用。"
          },
        ]
      },
    ]
  },
  { id: 7,
    name: "Tensor Reproducibility",
    sections: [
      { id: 0, 
        name: "7. 张量可重复性",
        value: "张量可重复性意味着在 PyTorch 中执行张量操作时, 即使多次运行相同的代码, 也能 consistently 获得相同的结果。这在机器学习中尤为关键, 特别是在模型训练过程中, 可靠的结果对于确保你的结果不是由于随机变化非常重要。",
        image: "",
        content: [
          { id: 1, 
            title: "",
            value: "机器学习算法通常涉及一些随机性。例如, 在初始化神经网络权重、将数据划分为训练集和测试集或在训练过程中打乱数据时, 这些过程通常涉及随机数生成。"
          }, 
          { id: 2, 
            title: "",
            value: "因此, 为了在 PyTorch 中实现张量的可重复性, 最常见的方法是设置随机种子。这有助于控制随机性, 并确保在不同的运行中获得一致的结果。"
          },
        ]
      }, 
      { id: 1, 
        name: "7.1 随机性",
        value: "True Randomness means that each possible outcome is equally likely, and there is no way to predict the next result based on previous ones.",
        image: "", 
        content: [
          { id: 1, 
            title: "", 
            value: "R随机性是指在一系列事件或结果中缺乏任何可预测的模式或顺序。在真正的随机过程中, 每个可能的结果的发生概率是相等的。例如, 当掷一枚公平的硬币时, 有 50% 的机会出现正面或反面, 但每次掷硬币的结果都是不可预测的。"
          }, 
        ]
      },
      { id: 2, 
        name: ":: PyTorch 中的随机性",
        value: "",
        image: "src/assets/chapter_one/randpy.jpeg", 
        content: [
          { id: 1, 
            title: "", 
            value: "随机性在许多 PyTorch 操作中发挥着重要作用。像 torch.rand() 这样的函数通常用于在张量中引入随机元素。torch.rand() 生成一个填充有从区间 [0, 1) 的均匀分布中抽取的随机数的张量。"
          }, 
          { id: 2, 
            title: "", 
            value: "了解这些函数的工作原理以及如何控制 PyTorch 中的随机性对于初始化模型参数、数据增强和执行随机操作等任务非常重要。"
          }, 
        ]
      },
      { id: 3, 
        name: ":: 随机性特征",
        value: "",
        image: "src/assets/chapter_one/randfeat.jpeg", 
        content: [
          { id: 1, 
            title: "不可预测性", 
            value: "在随机过程中, 未来的结果不能根据过去的事件来确定。例如, 当掷一枚公平的硬币时, 每次掷硬币都是独立的, 每次都有 50% 的机会得到正面或反面, 与以前的结果无关。"
          }, 
          { id: 2, 
            title: "真实随机性",
            value: "真实随机性通常源自自然过程, 如放射性衰变或热噪声——这些现象本质上是不可预测的, 并且不受以前事件的影响。"
          },
          { id: 3, 
            title: "随机性与伪随机性",
            value: "与伪随机性不同, 真实随机性是通过算法生成的, 如果知道种子, 则可以重现。虽然伪随机性可以模拟随机性, 但它具有潜在的确定性模式, 而真实随机性则无法精确重现。"
          }
        ]
      },
      { id: 4, 
        name: "7.2 伪随机性",
        value: "There is no real randomness in computers because the randomness is simulated. It's designed, so each step is predictable.",
        image: "", 
        content: [
          { id: 1, 
            title: "", 
            value: "伪随机性指的是那些看起来像是随机的序列, 但实际上是通过使用伪随机数生成器( PRNGs) 的确定性过程生成的。这些序列依赖于一个称为种子的初始值。"
          }, 
        ]
      },
      { id: 5, 
        name: ":: PSEUDORANDOMNESS Features",
        value: "",
        image: "src/assets/chapter_one/pseudorandf.jpeg", 
        content: [
          { id: 1, 
            title: "确定性", 
            value: "伪随机序列可能看起来是随机的, 但它们是由一个特定的、可重复的算法产生的。如果你知道种子和算法, 你可以准确预测或复制该序列。"
          }, 
          { id: 2, 
            title: "可重复性", 
            value: "伪随机性允许在计算任务中实现可重复性。通过使用相同的种子, 你可以生成相同的“随机”数列, 这对于调试、测试和科学研究至关重要。"
          }, 
          { id: 3, 
            title: "不完全随机", 
            value: "伪随机数是由算法生成的, 因此它们不是真正的随机数。即使这种模式不是立即显而易见的, 它们仍然具有潜在的模式或结构。另一方面, 真正的随机性将是完全不可预测的, 并且缺乏任何确定性的模式。"
          }, 
        ]
      },
      { id: 6, 
        name: "7.3 RANDOM SEED",
        value: "A random seed is used to perform repeatable experiments by reducing randomness in neural networks within PyTorch.",
        image: "", 
        content: [
          { id: 1, 
            title: "", 
            value: "So how can we achieve tensor reproducibility in PyTorch? The most common approach is to set random seeds."
          }, 
          { id: 2, 
            title: "", 
            value: "To ensure that the random number generators produce the same sequence of numbers each time you run the code, you should set the seed for PyTorch, NumPy, and Python's random module."
          }, 
          { id: 3, 
            title: "", 
            value: "Setting a random seed ensures that these random processes produce the same results every time you run your code. This is essential for debugging, sharing your work with others, or just making sure your model behaves consistently."
          }, 
        ]
      },
      { id: 7, 
        name: ":: PyTorch 中的随机种子",
        value: "",
        image: "src/assets/chapter_one/randseedpy.jpeg", 
        content: [
          { id: 1, 
            title: "设置随机种子为 69", 
            value: "从数学角度来看, 42 只是一个普通的数字, 用作种子时没有任何特别的性质使其在生成随机数时更好或更差。"
          }, 
          { id: 2, 
            title: "可以使用其他数字吗？", 
            value: "当然可以！你可以将种子设置为任何整数。选择 42 纯粹是惯例, 对生成随机数的质量没有影响。重要的是设置了种子, 以确保可重复性, 而不是使用的具体数字。"
          }, 
        ]
      },
    ]
  },
  { id: 8, 
    name: "Tensor on GPUs",
    sections: [
      { id: 0, 
        name: "8. Tensor on GPUs",
        value: "Running tensors on GPUs can significantly speed up computation, especially when dealing with large tensors or training deep learning models.",
        image: "", 
        content: [
          { id: 1, 
            title: "", 
            value: "在 PyTorch 中在 GPU 上运行张量涉及将张量数据传输到 GPU, 然后使用 GPU 加速进行操作。这可以大大加快计算速度, 尤其是在处理大型张量和训练深度学习模型时。"
          }
        ]
      }, 
      { id: 1, 
        name: "8.1 在 GPU 上运行张量", 
        value: "要在 GPU 上运行 PyTorch 张量, 你需要将张量数据移动到 GPU, 然后使用 GPU 加速进行操作。这确实可以加快计算速度, 特别是在处理大型张量和深度学习模型时。", 
        image: "src/assets/chapter_one/gpu_run.jpeg", 
        content: [
          { id: 1, 
            title: "", 
            value: "首先, 你需要检查机器上是否有可用的 GPU, 然后再将张量移动到 GPU 上。编写可以在 CPU 和 GPU 上运行的代码, 而无需修改。这使得你的代码更加灵活, 可以在可用的情况下使用 GPU 加速, 但如果没有 GPU 时仍然可以在 CPU 上运行。一旦确认 GPU 可用, 你可以使用 .to(device) 方法将张量移动到 GPU 上。如果你需要将张量移回 CPU( 例如, 在将其转换为 NumPy 数组之前) , 可以很容易地完成此操作。"
          },
        ]
      },
      { id: 2, 
        name: "8.2 GPUs",
        value: "A GPU, short from Graphics Processing Unit, is much faster at tensor computations compared to a CPU.",
        image: "",
        content: [
          { id: 1, 
            title: "", 
            value: "GPU( 图形处理单元) 是一种专门的处理器, 主要用于渲染图形和执行复杂的并行计算。虽然最初是为了满足游戏和多媒体的高计算需求而开发的, 但由于其高效处理大规模计算的能力, GPU 现在在科学计算、人工智能( AI) 和机器学习等领域变得不可或缺。"
          }
        ]
      },
      { id: 3, 
        name: ":: GPU 特性",
        value: "",
        image: "src/assets/chapter_one/gpu_feature.jpeg", 
        content: [
          { id: 1, 
            title: "并行处理", 
            value: "GPU 设计用于同时处理成千上万的任务, 使其在处理可以并行化的操作时非常理想, 例如图像处理、模拟和神经网络训练。",
          }, 
          { id: 2, 
            title: "核心结构",
            value: "与通常只有几个强大核心以优化顺序处理的 CPU 不同, GPU 具有数千个较小的核心, 可以同时处理多个任务。这种架构使得 GPU 在某些计算类型上比 CPU 快得多。",
          },
          { id: 3, 
            title: "能效",
            value: "虽然 GPU 功能强大, 但它们的功耗也很高, 特别是在执行像游戏或 AI 模型训练等要求苛刻的任务时。然而, 它们同时处理大量任务的能力使得它们在这些特定应用中往往比 CPU 更具能效。",
          },
          { id: 4, 
            title: "应用",
            value: "GPU 对实时渲染高质量图形至关重要, 这对现代视频游戏至关重要。它们还通过并行处理大量数据来加速神经网络的训练。此外, GPU 还用于模拟、分子建模、天气预报和其他需要快速处理大量数据的任务。",
          },
        ]
      }, 
      { id: 4, 
        name: "8.3 CUDA",
        value: "",
        image: "src/assets/chapter_one/gpu_cuda.jpg", 
        content: [
          { id: 1, 
            title: "", 
            value: "CUDA( 计算统一设备架构) 是由 NVIDIA 开发的并行计算平台和编程模型。它使开发者能够利用 NVIDIA GPU 的强大功能, 用于不仅仅是图形渲染的任务, 使 GPU 可用于通用计算。"
          }
        ]
      },
      { id: 5, 
        name: ":: CUDA 特性",
        value: "", 
        image: "src/assets/chapter_one/gpu_cuda_feature.jpeg", 
        content: [
          { id: 1, 
            title: "并行计算框架", 
            value: "CUDA 使开发者能够编写在 NVIDIA GPU 上运行的程序, 充分利用其强大的并行处理能力。这意味着计算可以比在 CPU 上进行得更快, 尤其是对于可以并行化的任务, 例如矩阵运算、模拟和神经网络训练。"
          },
          { id: 2, 
            title: "编程模型", 
            value: "CUDA 在标准编程语言如 C、C++ 和 Python 的基础上添加了用于 GPU 并行执行的特殊关键字和函数。这允许开发者编写在 CPU( 主机) 和 GPU( 设备) 上运行的代码, 其中 GPU 处理可以并行运行的计算部分。",
          }, 
          { id: 3, 
            title: "架构",
            value: "在 CUDA 中, GPU 作为主 CPU 的协处理器。典型的 CUDA 程序将问题分解为更小的子问题, 这些子问题可以由在 GPU 核心上运行的数千个轻量级线程并行处理。"
          },
          { id: 4, 
            title: "优势",
            value: "CUDA 利用 GPU 的并行特性大大加快了计算密集型任务的速度。它支持跨多个 GPU 的扩展以获得更大的计算能力。此外, 还提供了丰富的工具、库和框架, 包括用于深度学习的 cuDNN 和用于线性代数的 cuBLAS, 使得开发更加容易。"
          },
        ]
      }, 
      { id: 6, 
        name: "8.4 获取 GPUs",
        value: "",
        image: "src/assets/chapter_one/gpu_get.jpeg", 
        content: [
          { id: 1, 
            title: "检查你的本地计算机是否有 GPU", 
            value: "如果你有一台现代计算机, 特别是游戏或工作站电脑, 你可能已经配备了 GPU。如果不确定如何检查, 可以询问 chatGPT 或进行快速在线搜索以获取指导。找到后, 确保安装必要的驱动程序和 CUDA 工具包( 如果你有 NVIDIA GPU) 。"
          }, 
          { id: 2, 
            title: "购买 GPU", 
            value: "如果你的计算机没有 GPU, 你可以购买一个。适合你的最佳 GPU 将取决于你的预算和你希望执行的任务。"
          }, 
          { id: 3, 
            title: "使用云服务", 
            value: "如果你没有本地 GPU 或需要更强大的 GPU, 云服务提供灵活的选项。你可以使用如 Amazon Web Services (AWS)、NVIDIA GPU Cloud (NGC) 或 Microsoft Azure 等平台。对于简单的选项, Google Colab 提供免费的 GPU 访问, 如 NVIDIA K80、T4 和 P100。你还可以升级到 Colab Pro, 获得更快的 GPU 和更长的运行时间。"
          }
        ]
      }, 
    ]
  }
]