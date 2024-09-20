import {
  prepare_create,
  prepare_split,
  prepare_visual,
  build_architecture,
  build_model,
  train_model,
  train_visual,
  train_loss,
  improve_model,
  improve_one,
  improve_one_architec,
  improve_one_build,
  improve_one_test,
  improve_one_loss,
  improve_two,
  improve_two_build,
  improve_two_test,
  improve_two_loss,
  improve_three,
  improve_three_build,
  improve_three_test,
  improve_three_loss,
  save_choose,
  save_model_3,
  save_load_model_3
} from '../assets/chapter_four'

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
        image: prepare_create,
        content: [
          { id: 1,
            title: "",
            value: "Unlike our first project, in this one, we use Scikit Learn Datasets to create a circular distribution dataset. You’ll see a visualization of the distribution points soon."
          },
        ]
      },
      { id: 2, 
        name: ":: Split DATA", 
        value: "",
        image: prepare_split,
        content: [
          { id: 1,
            title: "",
            value: "Just like with the previous model, we split the data into a training set and a test set using the usual percentage. Here, we allocate 40 data points (80% of the total) to the training set, while the remaining 10 data points (20%) are reserved for the test set."
          },
        ]
      },
      { id: 3, 
        name: ":: Visualize DATA", 
        value: "",
        image: prepare_visual,
        content: [
          { id: 1,
            title: "",
            value: "Here we plot our dataset, showing that the 1,000 points are clearly distributed into two distinct clusters, each with a circular shape, with one cluster nested inside the other. In the next steps, we’ll build a model to classify these circular-shaped clusters."
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
            value: "This step involves selecting the appropriate model architecture that best suits the problem at hand. Once the architecture is defined, the model is implemented using a framework like PyTorch. This involves setting up the model’s layers, specifying activation functions, and initializing weights. The model is then ready to be trained on the data, where it will adjust its parameters to minimize errors and improve its predictions."
          },
        ]
      },
      { id: 1, 
        name: ":: Construct model Architecture", 
        value: "",
        image: build_architecture,
        content: [
          { id: 1,
            title: "",
            value: "To better understand this model, I visualized its architecture. The diagram clearly shows the structure: a simple deep learning model with three main components—input, a single hidden layer, and output. The input layer takes in two features, the hidden layer consists of five neurons, and the output layer produces a single feature. That’s all there is to it!"
          }
        ]
      },
      { id: 2, 
        name: ":: Build a Model with PyTorch", 
        value: "",
        image: build_model,
        content: [
          { id: 1,
            title: "",
            value: "First, we create a class for our neural network by inheriting from PyTorch’s base class. We define two layers: layer_1, which takes 2 input features and outputs 5 features, and layer_2, which reduces these 5 features to 1. Both layers use a linear function."
          },
          { id: 2,
            title: "",
            value: "Next, we override the forward() method to specify how the input data X moves through the network. The input goes through layer_1, then through layer_2, and the final output is returned."
          },
          { id: 3,
            title: "",
            value: "Finally, we instantiate the CircleModelV0 class and move the model to the desired device (CPU or GPU) for execution. This ensures the model runs on the specified hardware."
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
        image: train_model,
        content: [
          { id: 0,
            title: "",
            value: "There are two main differences between the previous model and this binary classification model: First, we introduce a new loss function — Binary Cross Entropy — suitable for binary classification tasks. Second, we utilize GPUs for training and testing data if available. Let's focsus on these two changes, there's no necessary to exaplain the whole steps again."
          },
          { id: 1,
            title: "Pick Up a New Loss Function",
            value: "We use Binary Cross Entropy, a popular loss function for measuring performance in binary classification problems."
          },
          { id: 2,
            title: "Put data on GPUs",
            value: "In this project, we’ll use GPUs for training and testing data if available."
          },
        ]
      },
      { id: 2, 
        name: ":: Test the trained Model", 
        value: "",
        image: train_visual,
        content: [
          { id: 1,
            title: "",
            value: "After training, we test our model on both the training and test data, resulting in two sets of predictions. When we plot these predictions, it’s clear that the model performs poorly. The plot shows a straight line with a slight angle separating the data points, regardless of whether they are from the training or test set."
          },
          { id: 2,
            title: "",
            value: "What a horribal predictions! This poor performance might be due to we used only linear functions in our model. Since the clusters are circular shape, a non-linear function could be more effective, as a straight line can’t properly separate circular shapes, right?"
          }
        ]
      },
      { id: 3, 
        name: ":: Check loss curves", 
        value: "",
        image: train_loss,
        content: [
          { id: 1,
            title: "",
            value: "Even though the model’s performance is poor, we should still examine the loss curves as part of the regular training process."
          },
          { id: 2,
            title: "",
            value: "From the loss curves, we see that both the training and test losses decrease smoothly and slowly as the number of epochs increases. We might want to consider increasing the number of epochs. However, the loss only dropped by about 0.01, from around 0.7 to 0.69, over 100 epochs. This slow progress suggests that even with 1,000 epochs, the model might not perform much better. But hey, it’s worth a shot to experiment, right?"
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
            value: "This can be done in several ways: adding more layers to increase the model’s depth, adding more hidden units to boost its capacity, increasing training epochs for more learning iterations, adjusting the learning rate for better convergence, changing the loss function, activation function or optimizer to better suit the problem, and using pre-trained models to leverage existing knowledge."
          },
        ]
      },
      { id: 1, 
        name: ":: How to Improve a Model", 
        value: "",
        image: improve_model,
        content: [
          { id: 0,
            title: "",
            value: "When we want to improve a model, we can look at two main scenarios: the model itself and the training process."
          },
          { id: 1,
            title: "Rebuild the Model",
            value: "The input and output layers are fixed, so we can focus on the hidden layers. We can add more layers to make it deeper or add more neurons to boost its capability. We can also think about changing the activation function (we’ll get into that in the math chapter). Using a pre-trained model is another option, which is really useful and something we’ll use in upcoming projects."
          },
          { id: 2,
            title: "Rebuild the Training Loop",
            value: "We can also train our model with different parameters and functions. For example, we can change the learning rate, set more training epochs, or try a different loss function or optimizer. In this model, we’ll only cover changing the learning rate and increasing the number of epochs."
          },
        ]
      },
    ]
  },
  { id: 4, 
    name: "The 1st Model (+ReLU)",
    sections: [
      { id: 0, 
        name: "5. The First Model (+ReLU)", 
        value: "",
        image: improve_one,
        content: [
          { id: 1,
            title: "",
            value: "Since we’ve used linear functions to build our model so far, we haven’t explored non-linear functions. Now, we’ll introduce our first non-linear formula as the activation function in this model to see how it performs."
          },
        ]
      },
      { id: 1, 
        name: ":: Architecture of Model", 
        value: "",
        image: improve_one_architec,
        content: [
          { id: 1,
            title: "",
            value: "Before we rebuild our model, let’s visualize its architecture with a new non-linear activation function - ReLU - a powerful non-linear function that eliminates all negative values flowing to neurons."
          }
        ]
      },
      { id: 2, 
        name: ":: Build and Train model", 
        value: "",
        image: improve_one_build,
        content: [
          { id: 1,
            title: "",
            value: "First, we add a new model parameter ‘relu’, which is the ReLU function. During the computation process, we place it between layer_1 and layer_2 to filter out the negative values from layer_1."
          }
        ]
      },
      { id: 3, 
        name: ":: Evaluate Model", 
        value: "",
        image: improve_one_test,
        content: [
          { id: 1,
            title: "",
            value: "After training and testing this new model, we plot the predictions against the training and test data. From this image, we can clearly see that our predictions are still poor, even with the addition of the non-linear function. Why is that? Do we need more training epochs? Let’s continue exploring."
          }
        ]
      },
      { id: 4, 
        name: ":: Check the Loss Curves", 
        value: "",
        image: improve_one_loss,
        content: [
          { id: 1,
            title: "",
            value: "Even though it’s still a poor model, we should check the loss curves—maybe we can uncover something and identify the key issue from another perspective."
          },
          { id: 2,
            title: "",
            value: "Comparing it to our original model, the curves change continuously, almost at the same pace as the epochs increase. The loss is still decreasing slowly, much like the original model."
          },
        ]
      },
    ]
  },
  { id: 5, 
    name: "The 2nd Model (+epochs)",
    sections: [
      { id: 0, 
        name: "6. The Second Model (+epochs)", 
        value: "",
        image: improve_two,
        content: [
          { id: 1,
            title: "",
            value: "In this model, we maintain the architecutre of the first model. However in the training loop, we increase the number of epochs from 100 to 1,000 for more learning iteration. Now, let's build this new model - model_2 - train it and then evaluate its performance."
          },
        ]
      },
      { id: 1, 
        name: ":: Build and Train Model", 
        value: "",
        image: improve_two_build,
        content: [
          { id: 1,
            title: "",
            value: "In this experiment, we keep the the model architecture the same as in model_1. Since we’re just rebuilding the same model, I didn’t include the code here - just refer to the previous one, with the model name updated, of course."
          },
          { id: 2,
            title: "",
            value: "This image shows the training loop settings, where the only change we made is increasing the epochs from 100 to 1,000."
          }
        ]
      },
      { id: 2, 
        name: ":: Evaluate Model", 
        value: "",
        image: improve_two_test,
        content: [
          { id: 1,
            title: "",
            value: "Looking at this plot, the predictions are pretty solid - much better than the last one. Almost 80% of the points are separated by the white curve, which is a great sign. It means we’re heading in the right direction."
          }
        ]
      },
      { id: 3, 
        name: ":: Check the Loss Curves", 
        value: "",
        image: improve_two_loss,
        content: [
          { id: 1,
            title: "",
            value: "From this image, we can see that the loss curves are smooth, with the loss values steadily decreasing as the training epochs progress. After halfway through, the curves start changing more rapidly up until the end of training. So, it’s likely that if we add more epochs, this line could continue to decrease at a similar pace."
          }
        ]
      },
    ]
  },
  { id: 6, 
    name: "The 3rd Model (->lr)",
    sections: [
      { id: 0, 
        name: "7. The Third Model (->lr)", 
        value: "",
        image: improve_three,
        content: [
          { id: 1,
            title: "",
            value: "Building on the first two changes we made in model_1 and model_2, now, let's focus on the learing rate - a crucial important hyperparameter in deep learing. Our first 2 models slowed slow changes in loss, so what if we increase the learning step. By taking a bigger step, we might get a better model with the same training epochs and same architecture. Let's experiment with this."
          },
        ]
      },
      { id: 1, 
        name: ":: Build and Train Model", 
        value: "",
        image: improve_three_build,
        content: [
          { id: 1,
            title: "",
            value: "Next, we’ll build a new model - model_3. In the training loop, we’ve adjusted the learning rate for a faster learning pace. Let’s train and evaluate this new model."
          }
        ]
      },
      { id: 2, 
        name: ":: Evaluate Model", 
        value: "",
        image: improve_three_test,
        content: [
          { id: 1,
            title: "",
            value: "Wow, from the plot, we can clearly see that our model successfully separates the two point clusters, even with the circular shape. The predictions are nearly 100% accurate—this is a perfect model!"
          }
        ]
      },
      { id: 3, 
        name: ":: Check the Loss Curves", 
        value: "",
        image: improve_three_loss,
        content: [
          { id: 1,
            title: "",
            value: "From these loss curves, we can clearly see that this model’s loss drops much faster to a lower level compared to the last model, even with the same number of training epochs. This shows just how crucial the learning rate is in our training process, we should pay more attention to this hyperparamater in future prejects."
          }
        ]
      },
    ]
  },
  { id: 7, 
    name: "Save a Model",
    sections: [
      { id: 0, 
        name: "8. Save a Model", 
        value: "Before we save the model, we need to choose the best one.",
        image: "",
        content: [
          { id: 1,
            title: "",
            value: "Well, as part of the regular model training process, the only different here is that we need to choose the best model to save, since we built several versions models to improve on the original model."
          },
        ]
      },
      { id: 1, 
        name: "5.0 Choose a Best Model", 
        value: "",
        image: save_choose,
        content: [
          { id: 1,
            title: "",
            value: "Here, we’ve plotted all the models—from the initial poor model to the perfect model_3—so you can clearly see the differences between them. Starting with model_0, you can see how the models progressively improve until the perfect one emerges. It’s clear, understandable, and visually striking."
          }
        ]
      },
      { id: 2, 
        name: "5.1 Save a Model", 
        value: "",
        image: save_model_3,
        content: [
          { id: 1,
            title: "",
            value: "Just follow the usual model-saving process, over!"
          }
        ]
      },
      { id: 3, 
        name: "5.2 Load a Model", 
        value: "",
        image: save_load_model_3,
        content: [
          { id: 1,
            title: "",
            value: "In the first step - preparing data - we load the data onto GPUs if available. However, we save our model on the CPUs. So, before making predictions with the saved model, we need to move this model to GPUs if they are available. That’s the only difference compared to the last project"
          }
        ]
      },
    ]
  },
]