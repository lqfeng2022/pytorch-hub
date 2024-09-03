export default [
  { id: 0, 
    name: "Prepare DATA",
    sections: [
      { id: 0, 
        name: "1. Prepare DATA", 
        value: "Preparing data is the essential initial step in machine learning that involves organizing and refining raw data to ensure it is suitable for training.",
        image: "",
        content: [
        ]
      },
      { id: 1, 
        name: "1.1 Create DATA", 
        value: "",
        image: "src/assets/chapter_four/prepare_create.jpeg",
        content: [
          { id: 1,
            title: "",
            value: "Different from our first project, here we’ll use Scikit Learn Datasets to create a circle distribution dataset, the method is make_circles(). Let’s see the first 5 data and plot it on the Cartesian coordinate system, and we use blue and orange to represent the label “1” and label “0”."
          },
        ]
      },
      { id: 2, 
        name: "1.2 Split DATA", 
        value: "Now, we’ve got some data, here we gotta split X and y into training set and test set.",
        image: "src/assets/chapter_four/prepare_split.jpeg",
        content: [
          { id: 1,
            title: "Training Set",
            value: "Training set is what the model learn from, and it’s usually be 80% of the total DATA. Here we use blue to mark the training set."
          },
          { id: 2,
            title: "Testing Set",
            value: "Testing set is what the model gets evaluated on, use to test what it has learned from the Training DATA, it’s about 20% of the total DATA. Here we use green to mark the testing set."
          }
        ]
      },
      { id: 3, 
        name: ":: Visualize Data", 
        value: "",
        image: "src/assets/chapter_four/prepare_visual.png",
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
    ]
  },
  { id: 1, 
    name: "Build a Model",
    sections: [
      { id: 0, 
        name: "2. Build a Model", 
        value: "Building a model is a crucial step in the machine learning process, where we design and implement a machine learning algorithm that can learn patterns from data.",
        image: "",
        content: [
          { id: 1,
            title: "",
            value: "This step involves selecting the appropriate model architecture that best suits the problem at hand. Once the architecture is defined, the model is implemented using a framework like PyTorch. This involves setting up the model’s layers, specifying activation functions, and initializing weights. The model is then ready to be trained on the data, where it will adjust its parameters to minimize errors and improve its predictions."
          },
        ]
      },
      { id: 1, 
        name: ":: Build a Model class", 
        value: "",
        image: "src/assets/chapter_four/build_model.jpeg",
        content: [
          { id: 1,
            title: "",
            value: "Just like our first project, first of all, we create a class that inherits from a base class for all neural network module in PyTorch. Then we initialize the layers instead of the parameters in the first project. In this class, we define two layers, layer_1 and layer_2. Layer_1 takes an input of 2 features and outputs 5 features, and layer_2 takes the 5 features from layer_1 and reduces them to 1 feature. And they are both done by a linear function in neural network."
          },
          { id: 2,
            title: "",
            value: "Then we overwrite the forward() method to define the forward pass of this network, describing how the input X is transformed as it passes through the network. The input X is first passed through layer_1, then the output of layer_1 is passed through layer_2. The result of layer_2 is returned as the output of the model."
          },
          { id: 3,
            title: "",
            value: "When we built a model, we gotta creates an instance of the CircleModelV0 class and transfers the model to the specified device (e.g., CPU or GPU). This step is necessary for running the model on a specific hardware device."
          },
        ]
      },
      { id: 2, 
        name: ":: Model Architecture", 
        value: "",
        image: "src/assets/chapter_four/build_architecture.jpeg",
        content: [
          { id: 1,
            title: "",
            value: "To get a clear understanding of this model, I visualized its architecture. From this diagram, we can easily see the structure of our model. It’s a simple deep learning model with three main components: the input, a single hidden layer, and the output. The input layer takes in two features, the hidden layer has five neurons, and the output layer produces a single feature. That’s pretty much it!"
          }
        ]
      },
    ]
  },
  { id: 2, 
    name: "Train a Model",
    sections: [
      { id: 0, 
        name: "3. Train a Model", 
        value: "Different from our former model, here we use a new loss function - Binary Cross Entropy, and put our training and testing data on GPUs if you have.",
        image: "",
        content: [
        ]
      },
      { id: 1, 
        name: ":: Build a training loop", 
        value: "",
        image: "src/assets/chapter_four/train_model.jpeg",
        content: [
          { id: 1,
            title: "Pick Up a Loss Function",
            value: "Here we use a new loss function - Binary Cross Entropy, which is a Popular Loss Function to Measure the Performance of a Binary Classification problem."
          },
          { id: 2,
            title: "Pick Up an Optimizer",
            value: "As the former model, here we choose Stochastic Gradient Descent, it's way efficient to compare with Gradient Descent."
          },
          { id: 3,
            title: "Set Up Training Epochs",
            value: "We set 100 epochs first of all, we can add more epochs in the improving model step if necessary."
          },
          { id: 4,
            title: "Set Up Training Loop",
            value: "Here's the training loop is exactly same as our first model, actually, these training steps are arranged perfectly, so we don't need to change it. And in the next projects, we're not gonna focus on these steps."
          },
          { id: 5,
            title: "Set Up Testing Loop",
            value: "Look at our testing loop, when we predict with our model, we use squeeze() method to delete an extra dimension. So everytime when we treate these data, we should take our eyes on their shape, is pretty important, you should be very very careful."
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
            value: "We use the testing data to make predictions with our trained model and plot the results as red points. From the image, we can see that these red points are very close to the green points, which represent the target values."
          }
        ]
      },
      { id: 3, 
        name: ":: Check loss curves", 
        value: "",
        image: "src/assets/chapter_four/train_loss.jpeg",
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
  { id: 3,
    name: "Improve a Model",
    sections: [
      { id: 0, 
        name: "4. Improve a Model", 
        value: "Saving a model involves storing the trained model’s parameters to a file, allowing you to reuse the model later without retraining it from scratch.",
        image: "",
        content: [
          { id: 1,
            title: "",
            value: "Saving a model is a critical step in the machine learning workflow. After training a model, you typically want to save it so that you can deploy it, share it, or resume training later without starting from scratch."
          },
        ]
      },
      { id: 1, 
        name: ":: How to Improve a Model", 
        value: "Saving a model involves storing the trained model’s parameters to a file, allowing you to reuse the model later without retraining it from scratch.",
        image: "src/assets/chapter_four/improve_model.jpeg",
        content: [
          { id: 1,
            title: "",
            value: "Saving a model is a critical step in the machine learning workflow. After training a model, you typically want to save it so that you can deploy it, share it, or resume training later without starting from scratch."
          },
        ]
      },
    ]
  },
  { id: 4, 
    name: "A better Model (+ReLU)",
    sections: [
      { id: 0, 
        name: "4.1 A better Model (+ReLU)", 
        value: "Saving a model involves storing the trained model’s parameters to a file, allowing you to reuse the model later without retraining it from scratch.",
        image: "src/assets/chapter_four/improve_one.jpeg",
        content: [
          { id: 1,
            title: "",
            value: "Saving a model is a critical step in the machine learning workflow. After training a model, you typically want to save it so that you can deploy it, share it, or resume training later without starting from scratch."
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
            value: "First, we create a directory for our model and save its state dictionary using the save() method in PyTorch."
          }
        ]
      },
      { id: 2, 
        name: ":: Build this model", 
        value: "",
        image: "src/assets/chapter_four/improve_one_build.jpeg",
        content: [
          { id: 1,
            title: "",
            value: " Later, when we need to use this model, we simply load the state dictionary back into the model using the load_state_dict() method."
          }
        ]
      },
      { id: 3, 
        name: ":: Train and Test Model", 
        value: "",
        image: "src/assets/chapter_four/improve_one_test.png",
        content: [
          { id: 1,
            title: "",
            value: " Later, when we need to use this model, we simply load the state dictionary back into the model using the load_state_dict() method."
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
            value: " Later, when we need to use this model, we simply load the state dictionary back into the model using the load_state_dict() method."
          }
        ]
      },
    ]
  },
  { id: 5, 
    name: "Another Model (+epochs)",
    sections: [
      { id: 0, 
        name: "4.2 Another Model (+epochs)", 
        value: "Saving a model involves storing the trained model’s parameters to a file, allowing you to reuse the model later without retraining it from scratch.",
        image: "src/assets/chapter_four/improve_two.jpeg",
        content: [
          { id: 1,
            title: "",
            value: "Saving a model is a critical step in the machine learning workflow. After training a model, you typically want to save it so that you can deploy it, share it, or resume training later without starting from scratch."
          },
        ]
      },
      { id: 1, 
        name: ":: Build this model", 
        value: "",
        image: "src/assets/chapter_four/improve_two_build.jpeg",
        content: [
          { id: 1,
            title: "",
            value: " Later, when we need to use this model, we simply load the state dictionary back into the model using the load_state_dict() method."
          }
        ]
      },
      { id: 2, 
        name: ":: Train and Test Model", 
        value: "",
        image: "src/assets/chapter_four/improve_two_test.png",
        content: [
          { id: 1,
            title: "",
            value: " Later, when we need to use this model, we simply load the state dictionary back into the model using the load_state_dict() method."
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
            value: " Later, when we need to use this model, we simply load the state dictionary back into the model using the load_state_dict() method."
          }
        ]
      },
    ]
  },
  { id: 6, 
    name: "A Best Model (->lr)",
    sections: [
      { id: 0, 
        name: "4.3 A Best Model (->lr)", 
        value: "Saving a model involves storing the trained model’s parameters to a file, allowing you to reuse the model later without retraining it from scratch.",
        image: "src/assets/chapter_four/improve_three.jpeg",
        content: [
          { id: 1,
            title: "",
            value: "Saving a model is a critical step in the machine learning workflow. After training a model, you typically want to save it so that you can deploy it, share it, or resume training later without starting from scratch."
          },
        ]
      },
      { id: 1, 
        name: ":: Build this model", 
        value: "",
        image: "src/assets/chapter_four/improve_three_build.jpeg",
        content: [
          { id: 1,
            title: "",
            value: " Later, when we need to use this model, we simply load the state dictionary back into the model using the load_state_dict() method."
          }
        ]
      },
      { id: 2, 
        name: ":: Train and Test Model", 
        value: "",
        image: "src/assets/chapter_four/improve_three_test.png",
        content: [
          { id: 1,
            title: "",
            value: " Later, when we need to use this model, we simply load the state dictionary back into the model using the load_state_dict() method."
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
            value: " Later, when we need to use this model, we simply load the state dictionary back into the model using the load_state_dict() method."
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
          { id: 3,
            title: "",
            value: "Saving a model ensures that the time and resources spent on training aren’t lost, and it also makes it easier to deploy the model in production environments where it can be used to make predictions on new data. Additionally, saved models can be shared with others or reused in different projects."
          },
        ]
      },
      { id: 1, 
        name: "4.0 Choose a Best Model", 
        value: "",
        image: "src/assets/chapter_four/save_choose.png",
        content: [
          { id: 1,
            title: "",
            value: "First, we create a directory for our model and save its state dictionary using the save() method in PyTorch."
          }
        ]
      },
      { id: 2, 
        name: "4.1 Save a Model", 
        value: "",
        image: "src/assets/chapter_four/save_model_3.jpeg",
        content: [
          { id: 1,
            title: "",
            value: "First, we create a directory for our model and save its state dictionary using the save() method in PyTorch."
          }
        ]
      },
      { id: 3, 
        name: "4.2 Load a Model", 
        value: "",
        image: "src/assets/chapter_four/save_load_model_3.jpeg",
        content: [
          { id: 1,
            title: "",
            value: " Later, when we need to use this model, we simply load the state dictionary back into the model using the load_state_dict() method."
          }
        ]
      },
    ]
  },
]