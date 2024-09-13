export default [
  { id: 0, 
    name: "PyTorch Workflow",
    sections: [
      { id: 0,
        name: "0. PyTorch Workflow", 
        value: "A PyTorch workflow typically follows a sequence of steps that guide you through building, training, and evaluating deep learning models.",
        image: "",
        content: [
          { id: 1, 
            title: "",
            value: "A PyTorch workflow is a structured approach to developing deep learning models using PyTorch. It starts with preparing your data, then involves constructing the model, training it with the data, and evaluating its performance."
          },
          { id: 2, 
            title: "",
            value: "After assessing the model, you can make improvements and finally save the model for later use. This sequence ensures a systematic process from data handling to model deployment."
          },
        ]
      },
      { id: 1,
        name: ":: Workflow Overview", 
        value: "",
        image: "src/assets/chapter_two/workflow.jpeg",
        content: [
          { id: 1, 
            title: "",
            value: "In a PyTorch machine learning workflow, you start by preparing the data, which includes collecting and preprocessing it for model training. Next, you build the model by defining its architecture and setting up the training parameters. The model is then trained and evaluated to measure its performance and make improvements as needed. Finally, the model is saved and loaded for future use, completing the workflow."
          },
        ]
      },
      { id: 2,
        name: ":: Workflow Details", 
        value: "",
        image: "src/assets/chapter_two/workflow_explain.jpeg",
        content: [
          { id: 1, 
            title: "Prepare DATA",
            value: "In our first project, we’ll be working with a simple straight line, so preparing the data is straightforward. We’ll generate linear data and then split it into training and testing sets."
          },
          { id: 2, 
            title: "Build a Model",
            value: "Building the model involves designing the neural network architecture. For this initial project, our model is as simple as it gets — a straight line, represented by a linear function with just two parameters. There are no hidden layers; data flows directly from input to output."
          },
          { id: 3, 
            title: "Train a Model",
            value: "Training and evaluating the model are combined in this step. As we build our training loop, we’ll also evaluate the model’s performance after each iteration as the parameters are updated. This process will follow the typical steps of training and testing a neural network."
          },
          { id: 4, 
            title: "Improve a Model",
            value: "After training, it’s crucial to assess the model’s performance. If the results are not satisfactory, improvements are necessary. Well, for our first model, we’ll skip this step. In future projects, we’ll dive deeper into this process, placing major emphasis on refining models to achieve better and more efficient outcomes."
          },
          { id: 5, 
            title: "Save a Model",
            value: "Once we’ve developed a good model, it’s important to save it for reuse or further training. In this case, we’ll primarily save the model’s parameters."
          },
        ]
      },
    ]
  },
  { id: 1, 
    name: "Prepare DATA",
    sections: [
      { id: 0, 
        name: "1. Prepare DATA", 
        value: "Preparing data is the first step in machine learning, involving the organization and refinement of raw data to ensure it is suitable for training.",
        image: "",
        content: [
          { id: 1,
            title: "",
            value: "Preparing data is the first step in machine learning that involves organizing and refining raw data to ensure it is suitable for training. The quality and structure of this data directly impact the performance and accuracy of the model, making this step crucial for the overall success of the project."
          },
        ]
      },
      { id: 1, 
        name: "1.1 Create DATA", 
        value: "",
        image: "src/assets/chapter_two/prepare_create.jpeg",
        content: [
          { id: 1,
            title: "",
            value: "In the data preparation stage, we start by creating the input data using torch.arange(0, 1, 0.02), which generates 50 evenly spaced numbers between 0 and 1. This sequence of values will serve as the input for our model."
          },
          { id: 2,
            title: "",
            value: "To generate the corresponding output data, we apply a simple linear formula:  y = 0.6x + 0.1 . This equation produces the target values that our model will learn to predict."
          }
        ]
      },
      { id: 2, 
        name: "1.2 Split DATA", 
        value: "",
        image: "src/assets/chapter_two/prepare_split.jpeg",
        content: [
          { id: 1,
            title: "",
            value: "Once the input and output data are generated, the next step is to split the data into training and testing sets. We allocate 40 data points (the 80% of total) to the training set, which the model will use to learn the relationship between the inputs and outputs."
          },
          { id: 2,
            title: "",
            value: "The remaining 10 data points (the 20% of total) are reserved for the testing set, allowing us to evaluate the model’s performance on data it hasn’t seen before. This split ensures that the model is trained effectively while also providing a means to assess its accuracy and generalization ability."
          }
        ]
      },
      { id: 3, 
        name: ":: DATA Visualization ", 
        value: "",
        image: "src/assets/chapter_two/prepare_visual.png",
        content: [
          { id: 1,
            title: "",
            value: "Here, we visualize our target model using the pyplot module from matplotlib. The resulting image clearly shows our model as a series of points arranged in a straight line, with the training data marked in blue and the testing data in green."
          },
          { id: 2,
            title: "",
            value: "This visualization provides a clear and effective way to display our data, making it easier to understand the model’s structure. We’ll be using this method frequently in the next steps and future projects."
          }
        ]
      },
    ]
  },
  { id: 2, 
    name: "Build a Model",
    sections: [
      { id: 0, 
        name: "2. Build a Model", 
        value: "Building a model involves designing and implementing a machine learning algorithm that learns patterns from data to make predictions or decisions.",
        image: "",
        content: [
          { id: 1,
            title: "",
            value: "Building a model is a crucial step in the machine learning process, where we design and implement a machine learning algorithm that can learn patterns from data. This step involves selecting the appropriate model architecture that best suits the problem at hand."
          },
          { id: 2,
            title: "",
            value: "Once the architecture is defined, the model is implemented using a framework like PyTorch. This involves setting up the model’s layers, specifying activation functions, and initializing weights. The model is then ready to be trained on the data, where it will adjust its parameters to minimize errors and improve its predictions."
          },
        ]
      },
      { id: 1, 
        name: ":: Build a Model class", 
        value: "",
        image: "src/assets/chapter_two/build_model.jpeg",
        content: [
          { id: 1,
            title: "",
            value: "When building a model, the first step is to define and initialize the parameters that the model will use to learn from the data. These parameters, such as weights and biases, are crucial as they determine how the model will process the input data."
          },
          { id: 2,
            title: "",
            value: "Once the parameters are set, we can then define the computational process, outlining how the data will flow through the model, from input to output, and how the model will use the parameters to make predictions."
          }
        ]
      },
      { id: 2, 
        name: ":: Model Architecture", 
        value: "",
        image: "src/assets/chapter_two/build_architecture.jpeg",
        content: [
          { id: 1,
            title: "",
            value: "The architecture of this model is quite simple, consisting of just an input and an output, connected by a linear function."
          }
        ]
      },
      { id: 3, 
        name: ":: Model Visualization", 
        value: "",
        image: "src/assets/chapter_two/build_visual.png",
        content: [
          { id: 1,
            title: "",
            value: "We plot our initial model using red points and compare it with the target model (green points). This allows us to clearly see the differences between the current model and the target one. As expected, there’s quite a noticeable difference."
          }
        ]
      },
    ]
  },
  { id: 3, 
    name: "Train a Model",
    sections: [
      { id: 0, 
        name: "3. Train a Model", 
        value: "Training a model involves optimizing its parameters using labeled data, while testing evaluates the model’s performance on unseen data to assess its generalization ability.",
        image: "",
        content: [
          { id: 1,
            title: "",
            value: "Training a model involves adjusting its parameters based on labeled data to minimize errors and improve accuracy."
          },
          { id: 2,
            title: "",
            value: "Testing evaluates the model’s performance on new, unseen data to measure how well it generalizes to different scenarios."
          },
        ]
      },
      { id: 1, 
        name: "3.1 Build a train loop", 
        value: "",
        image: "src/assets/chapter_two/train_model.jpeg",
        content: [
          { id: 1,
            title: "Pick Up a Loss Function",
            value: "This is the first crucial step in model training, as the loss function quantifies how well the model’s predictions match the actual labels."
          },
          { id: 2,
            title: "Pick Up an Optimizer",
            value: "The optimizer is responsible for updating the model’s parameters based on the gradients of the loss function. Choosing the right optimizer (like SGD, Adam, etc.) can significantly affect the model’s convergence and performance."
          },
          { id: 3,
            title: "Set Up Training Epochs",
            value: "Defining the number of epochs (iterations over the entire dataset) is important to ensure the model has enough time to learn but not so much that it overfits. This step involves balancing training duration and model performance."
          },
          { id: 4,
            title: "Set Up Training Loop",
            value: "The training loop is where the actual learning happens. It iteratively feeds batches of data to the model, computes the loss, and updates the parameters using the optimizer. This step implements the core of the training process."
          },
          { id: 5,
            title: "Set Up Testing Loop",
            value: "After training, the testing loop evaluates the model on unseen data. It helps assess the model’s generalization ability by checking its performance on a separate test set."
          },
        ]
      },
      { id: 2, 
        name: ":: Visualize the Training Model", 
        value: "",
        image: "src/assets/chapter_two/train_visual_before.jpeg",
        content: [
          { id: 1,
            title: "",
            value: "We’ve already plotted the initial model, so now let’s focus on visually understanding how to train it. The training process is actually quite straightforward. It mainly involves two key steps: calculating the loss and then using the optimizer. Essentially, we repeat these two steps over and over until we find the optimal parameters for the model."
          }
        ]
      },
      { id: 3, 
        name: ":: Visualize the Trained Model", 
        value: "",
        image: "src/assets/chapter_two/train_visual_after.jpeg",
        content: [
          { id: 1,
            title: "",
            value: "After completing 100 epochs of training, we plot our model again. From the image, we can see that the difference between our model and the target model is now quite small. Compared to the initial model, our current model is much closer to the target."
          }
        ]
      },
      { id: 4, 
        name: ":: Test the trained Model", 
        value: "",
        image: "src/assets/chapter_two/train_visual_testd.jpeg",
        content: [
          { id: 1,
            title: "",
            value: "We use the testing data to make predictions with our trained model and plot the results as red points. From the image, we can see that these red points are very close to the green points, which represent the target values."
          }
        ]
      },
      { id: 5, 
        name: ":: Check loss curves", 
        value: "Loss curves display how the model’s error decreases over time during training, with separate lines for training and testing data. Ideally, both curves should show a downward trend, indicating that the model is learning and generalizing well.",
        image: "src/assets/chapter_two/train_loss_curves.jpeg",
        content: [
          { id: 1,
            title: "Train Loss (Blue Line)",
            value: "The training loss decreases steadily as the epochs progress, indicating that the model is learning and improving its performance on the training data. The curve flattens out towards the end, suggesting that the model is nearing convergence."
          },
          { id: 2,
            title: "Test Loss (Orange Line)",
            value: "The testing loss also decreases over time but starts higher than the training loss and remains above it throughout the training process. This suggests that the model is generalizing reasonably well but still performs slightly worse on the unseen test data compared to the training data."
          },
          { id: 3,
            title: "Overall",
            value: "Both curves decreasing is a good sign, indicating that the model is improving during training and is not overfitting, as the test loss is also lowering over time. The gap between the two curves could be further minimized with additional tuning."
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
            value: "Saving a model is a critical step in the machine learning workflow. After training a model, you typically want to save it so that you can deploy it, share it, or resume training later without starting from scratch."
          },
          { id: 2,
            title: "",
            value: "The saved model includes the learned parameters (weights and biases) and sometimes the model’s architecture, making it possible to reload and use the model immediately for making predictions."
          },
        ]
      },
      { id: 1, 
        name: "4.1 Save a Model", 
        value: "",
        image: "src/assets/chapter_two/save_model.jpeg",
        content: [
          { id: 1,
            title: "",
            value: "First, we create a directory for our model and save its state dictionary using the save() method in PyTorch."
          }
        ]
      },
      { id: 2, 
        name: "4.2 Load a Model", 
        value: "",
        image: "src/assets/chapter_two/save_model_load.jpeg",
        content: [
          { id: 1,
            title: "",
            value: " Later, when we need to use this model, we simply load the state dictionary back into a new model (initial with random weight) using the load_state_dict() method."
          }
        ]
      },
    ]
  },
]