export default [
  { id: 0, 
    name: "Classification Problem",
    sections: [
      { id: 0,
        name: "1. Classification Probelem", 
        value: "A classification problem involves categorizing an object into one of n distinct classes.",
        image: "",
        content: [
          { id: 1, 
            title: "",
            value: "A classification problem is a type of machine learning problem where the goal is to assign an input to one of several predefined categories or classes. Basically, we're analyzing an object or data point and placing it into one of n possible classes based on its features."
          },
          { id: 2, 
            title: "",
            value: "For example, in a binary classification problem, we only have two possible categories, like ‘spam’ or ‘not spam’ in an email filter. For multi-class classification, there can be several options, like deciding if an image is of a dog, cat, or bird. The main goal of the classification algorithm is to learn from the training data and make good predictions on new, unseen data by generalizing from the examples it has already learned."
          },
        ]
      },
      { id: 1,
        name: "1.1 Classification Probelm", 
        value: "",
        image: "src/assets/chapter_five/classific.jpeg",
        content: [
          { id: 1, 
            title: "Binary Classification",
            value: "Binary classification is the simplest form of classification, where an input is sorted into one of two distinct classes. It’s basically a decision-making process with just two possible outcomes. A common example would be spam detection, where emails are classified as either “spam” or “not spam”."
          },
          { id: 2, 
            title: "Multi-Class Classification",
            value: "Multi-class classification is about classify an input into one of more than two classes. Unlike binary classification, which has just two options, multi-class classification handles problems with multiple categories. A good example is image recognition, where image are classified as “cat”, “dog” or “bird” among other possibilites."
          },
          { id: 3, 
            title: "Multi-Label Classification",
            value: "Multi-label classification allows an input to be assigned multiple labels. Unlike binary and multi-class classification, where each input is tied to just one class, in multi-label classification, an input can belong to several classes at once. A good example is movie genre classification, where a movie can be labeled as “action”, “comedy” and “thriller” all at the same time."
          },
        ]
      },
      { id: 2,
        name: "1.2 Binary Classification", 
        value: "Binary classification is a fundamental task that assigns inputs to one of two distinct categories.",
        image: "",
        content: [
          { id: 1, 
            title: "Simplicity",
            value: "Binary classification is the simplest form of classification, involving just two classes. This simplicity makes it easier to understand and implement compared to more complex tasks."
          },
          { id: 2, 
            title: "Building Block",
            value: "Many complex machine learning tasks, like multi-class and multi-label classification, can be viewed as extensions or combinations or combinations of binary classification problems. For example, multi-class classification can often be broken down into several binary classification tasks."
          },
          { id: 3, 
            title: "Wide Application",
            value: "Binary classification has numerous real-world applications, including spam detection, disease diagnosis, and sentiment analysis. These tasks are common across various industries, making binary classification a key component of many machine learning systems."
          },
          { id: 4, 
            title: "Algorithm Foundation",
            value: "Many machine learning algorithms are initially developed and tested on binary classification tasks before being extended to handle more complex scenarios. Understanding binary classification is essential for grasping the core concepts of these algorithms."
          },
        ]
      },
      { id: 3,
        name: ":: Linear Relationship", 
        value: "",
        image: "src/assets/chapter_five/classific_linear.jpeg",
        content: [
          { id: 1, 
            title: "",
            value: "A linear relationship means that the two classes can be separated by a straight line in 2D, a plane in 3D, or a hyperplane in higher dimensions."
          },
          { id: 2, 
            title: "",
            value: "Here let’s focus on the simplest one, a straight line. A linear model is designed to find a straight line that best separates the two classes."
          },
          { id: 3, 
            title: "",
            value: "Linear models are straightforward and easy to interpret, often faster to train and require fewer resources. However, there have limitations, such as struggling with data that is not linearly separable. If the decision boundary between the two classes is curved or complex, a linear model may not perform well.",
          }
        ]
      },
      { id: 4,
        name: ":: Non-Linear Relationship", 
        value: "",
        image: "src/assets/chapter_five/classific_nonlinear.jpeg",
        content: [
          { id: 1, 
            title: "",
            value: "A non-linear relationship means that the classes cannot be separated by a straight line, a plane or hyperplane. Instead, the decision boundary may take the form of a curve or a more complex shape."
          },
          { id: 2, 
            title: "",
            value: "In non-linear relationships, the decision boundary could be a curve, multiple curves, or even more complex shapes."
          },
          { id: 3, 
            title: "",
            value: "While non-linear model offer greater flexibility and can capture complex patterns and relationships, making them suitable for a border range of problems. They are generally more complex and harder to interpret. They also lead to require more computational resources and may take longer to train."
          },
        ]
      },
    ],
  },
  { id: 1, 
    name: "Loss Function - Binary Cross Entropy",
    sections: [
      { id: 0,
        name: "2. Binary Cross Entropy", 
        value: "Binary cross-entropy is a popular loss function to measure the performance of binary classification problems.",
        image: "",
        content: [
          { id: 1, 
            title: "",
            value: "Binary cross-entropy is a widely used loss function in machine learning for binary classification tasks. It measures the difference between the predicted probabilities and the actual binary labels (0 or 1), quantifying how well the model’s predictions align with the true labels."
          },
          { id: 2, 
            title: "",
            value: "In binary classification, the goal is to assign each input to one of two possible classes. Binary cross-entropy evaluates model accuracy by penalizing predictions that diverge from the true labels. A low cross-entropy loss indicates that the predicted probability is close to the actual class, reflecting good model performance, while a high loss suggests significant deviation and poor performance."
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
            value: "Cross-entropy might sound fancy, but it's quite straightforward in binary classification. When dealing with two classes, we usually set the true label to 1. The Binary Cross Entropy formula then simplifies to BCE = -log(x), where x is the probability of the prediction being correct, ranging from 0 to 1."
          },
          { id: 2, 
            title: "",
            value: "If you check the figure showing the relationship between the predicted probability (x) and the training loss, you'll see this concept in action."
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
            value: "Cross-entropy measures the difference between two probability distributions: the true distribution P and an estimated distribution Q. Regardless of the number of classes. It evaluates how wll one probability distribution approximates another. The term “cross” in “cross-entropy” highlights the idea of comparing or combining these two different distributions."
          },
          { id: 2, 
            title: "",
            value: "In binary classification, where there are only two possible classes, binary cross-entropy is a specific form of cross-entropy that deals with the case of two distributions—one for each class."
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
            value: "The concept of entropy was first introduced in thermodynamics to measure the level of disorder or unpredictability in a system."
          },
          { id: 2, 
            title: "",
            value: "In information theory, entropy quantifies the uncertainty or randomness in a set of data. It quantifies how much information is gained when data is split or categorized, making it a crucial concept in both data science and machine learning."
          },
        ],
      },
    ],
  },
  { id: 2, 
    name: "Sigmoid Function",
    sections: [
      { id: 0,
        name: "3. Sigmoid Function", 
        value: "The sigmoid function takes any real-valued input and maps it to a value between 0 and 1.",
        image: "",
        content: [
          { id: 1, 
            title: "",
            value: "The sigmoid function is useful because it converts any input into a probability-like output, making it ideal for classification tasks."
          },
          { id: 2, 
            title: "",
            value: "In our binary classification model, we use the sigmoid function in conjunction binary cross-entropy. First, we convert the input values into a range between 0 and 1 using the sigmoid function. We then pass these values into the binary cross-entropy function to calculate the loss. We do this because binary cross-entropy is designed to work with probability-like values, which fall between 0 and 1."
          },
        ]
      },
      { id: 1,
        name: ":: Sigmoid Function Formula", 
        value: "",
        image: "src/assets/chapter_five/sigmoidFormula.jpeg",
        content: [
          { id: 1, 
            title: "S-shape",
            value: "Its graph has an S-shape: it starts near 0, rises steeply in the middle, and levels off as it approaches 1."
          },
          { id: 2, 
            title: "Output: 0 ~ 1",
            value: "The sigmoid function always outputs values between 0 and 1, making it ideal for representing probabilities."
          },
          { id: 3, 
            title: "Point: (0, 0.5)",
            value: "Symmetric around x=0, where the output is 0.5."
          },
          { id: 4, 
            title: "Applications",
            value: "The sigmoid function is crucial in scenarios where decisions based on the probability of an event occurring, such as determining an email as spam or not."
          },
        ]
      },
      { id: 2,
        name: ":: Sigmoid Function Features", 
        value: "",
        image: "src/assets/chapter_five/sigmoidFeature.jpeg",
        content: [
          { id: 1, 
            title: "Probability Output",
            value: "The sigmoid function maps a broad input values to a range between 0 and 1. This allows us to convert diverse input into a probability distribution."
          },
          { id: 2, 
            title: "Love Binary Classification",
            value: "The sigmoid function is particularly valuable for binary classification tasks because it compresses input values into the [0, 1] range, effectively handling large inputs."
          },
          { id: 3, 
            title: "Vanishing Graidien Issue",
            value: "While the sigmoid function has a simple gradient calculation, the gradient can become very small for extreme input values. This leads to minimal changes in weights during training, a problem known as the vanishing gradient issue."
          },
          { id: 4,
            title: "Computationally Expensive",
            value: "The sigmoid function involves an exponential calculation, making it computationally exprensive and less efficient compared to some other activation functions.",
          }
        ]
      },
    ],
  },
  { id: 3,
    name: "BackPropagation",
    sections: [
      { id: 0,
        name: "4. BackPropagation",
        value: "Backpropagation is a key algorithm for training neural networks, which involves the 'backward propagation of errors' to adjust weights and minimize loss.",
        image: "",
        content: [
          { id: 0, 
            title: "",
            value: "Backpropagation is a neural network training process by feeding the error rates from the predictions (a forward propagation) back through the network to adjust the parameters like weights, improving the model's accuracy."
          },
          { id: 1, 
            title: "Forward Propagation",
            value: "During forward propagation, the network generates predictions, and the difference between these predictions and the actual target values is calculated as the error."
          },
          { id: 2, 
            title: "Backward Propagation",
            value: "The error is then propagated backward through the network, layer by layer. In this process, the weights of the connections between neurons are adjusted to minimize the error."
          },
          { id: 3, 
            title: "Loop: Forward -> Backword",
            value: "By continuously updating weights based on the error, the network iteratively learns and enhn]ances its accuracy, progressively reducing the discrepancy between its predictions and the actual values."
          },
        ]
      },
      { id: 1,
        name: ":: Forward Propagation:",
        value: "",
        image: "src/assets/chapter_five/backpropagat_implem.jpeg",
        content: [
          { id: 0, 
            title: "",
            value: "To demonstrate the backpropagation concept, I’ve drawn a simple neural network with two hidden layers. This diagram outlines the network and shows the forward propagation flow, where input data pass through the network to generate predicted output."
          },
        ]
      },
      { id: 2,
        name: ":: Backward Propagation:",
        value: "",
        image: "src/assets/chapter_five/backpropagat_calcul.jpeg",
        content: [
          { id: 0, 
            title: "",
            value: "The main goal of backpropagation is to update the weights and biases to minimize the error or los between the predicted output and the desired output. The process involves:"
          },
          { id: 1, 
            title: "Calculate the Loss",
            value: "The loss is typically measured using functions like Mean Squared Error (MSE)."
          },
          { id: 2, 
            title: "Compute the Gradients",
            value: "The key step in backpropagation is computing gradients of the loss function with respect to the weights and biases. Using the chain rule, we calculate how much each weight and bias contributes to the overall error, which involves computing partial derivatives."
          },
          { id: 3, 
            title: "Update the Weights and Biases",
            value: "After computing the gradients, the weights and biases are updated to reduce the error. This is typically done using gradient descent, as coverd in the previous model."
          },
          { id: 4, 
            title: "Repeat the Process",
            value: "The forward and backward propagation steps are repeated iteratively until the model converges, meaning the loss function reaches a minimum, and the network accurately predicts the output for the given inputs."
          },
        ]
      },
    ]
  },
  { id: 4, 
    name: "ReLU Function",
    sections: [
      { id: 0,
        name: "5. ReLU Function", 
        value: "ReLU (Rectified Linear Unit) is a non-linear activation function commonly used in deep learning.",
        image: "",
        content: [
          { id: 1, 
            title: "",
            value: "The ReLU function is widely used in deep learning, especially in convolutional neural networks (CNNs) and various other neural network architectures."
          },
        ]
      },
      { id: 1,
        name: ":: ReLU Function Formula", 
        value: "",
        image: "src/assets/chapter_five/reluFormula.jpeg",
        content: [
          { id: 1, 
            title: "Output: (0, x)",
            value: "The ReLU function has a staightforward formula: it outputs the maximum value between 0 and input value x. As a result, the output of ReLU is always non-negative, ranging from 0 to infinity. The function is linear for positive inputs and outputs 0 for negative inputs."
          },
          { id: 2, 
            title: "Overall",
            value: "ReLU is favored in many deep learning models because of its simplicity, efficiency, and its ability to enable faster and more effective training of deep neural networks."
          },
        ]
      },
      { id: 2,
        name: ":: ReLU Function Features", 
        value: "",
        image: "src/assets/chapter_five/reluFeature.jpeg",
        content: [
          { id: 1, 
            title: "Non-linearity",
            value: "Although ReLU is simple and piecewise linear, it introduces non-linearity into the network, which is essential for learning complex patterns."
          },
          { id: 2, 
            title: "Computationally Efficient",
            value: "ReLU activates only neurons with positive inputs, resulting in sparse activations. This sparsity can enhance computational efficiency."
          },
          { id: 3, 
            title: "No Vanishing Gradient Issue",
            value: "Unlike sigmoid or tanh functions, ReLU avoid the vanishing gradient problem for positive inputs, enabling faster training and better performance in deep networks."
          },
          { id: 4, 
            title: "Dying ReLU issue",
            value: "However, if too many inputs are negative, neurons can become inactive and output 0, effectively ceasing to learn. This issue is known as the “dying ReLU” problem."
          },
        ]
      },
    ],
  },
]