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
            title: "A straight-line Relationship",
            value: "Linear regression is a statistical method that models a straight-line relationship between variables. It seeks to fit a line that best represents the data, minimizing the differences between observed and predicted values. This approach is widely used for making predictions and understanding trends in various fields."
          },
          { id: 2, 
            title: "Two types of Variables",
            value: "In its simplest form, linear regression involves two types of variables: an independent variable and a dependent variable. The independent variable is used to predict the outcome of the dependent variable. This method can also be extended to include multiple independent variables, enhancing the model’s predictive power."
          },
          { id: 3, 
            title: "Linear Regression in Machine Learning",
            value: "In machine learning, linear regression both a foundational algorithm and a gateway to more complex models. It is often one of the first techniques taught because it's straightforward and easy to interpret, making it an essential tool for understanding the basics of predictive modeling. In machine learning, linear regression is used to predict continuous outcomes, such as prices, temperatures, or scores, based on input data."
          },
        ]
      },
      { id: 1,
        name: ":: Linear Relationship", 
        value: "",
        image: "src/assets/chapter_three/linear.jpeg",
        content: [
          { id: 1, 
            title: "",
            value: "Here’s an example to visualize a simple linear relationship between working hours and salary. The blue line shows that the salary is directly proportional to the number of working hours, with a rate of $10 per hour."
          },
          { id: 2, 
            title: "",
            value: "This straight line illustrates how each additional hour worked results in a $10 increase in salary, making it a straightforward linear regression model. This model effectively captures and predicts outcomes based on the linear relationship between these two variables."
          },
        ]
      },
    ]
  },
  { id: 1, 
    name: "Normal Distribution",
    sections: [
      { id: 0,
        name: "2. Normal Distribution", 
        value: "A Normal Distribution is a type of continuous probability distribution for a real-valued random variable, typically forming a bell curve.",
        image: "",
        content: [
          { id: 1, 
            title: "What is Normal distribution?",
            value: "A Normal Distribution is a type of continuous probability distribution for a real-valued random variable. In simpler terms, it’s a way to describe how certain types of data naturally tend to spread out. When data follows a normal distribution, most of the values cluster around a central point, known as the mean, with fewer values appearing as you move further away from that center. This creates the classic “bell-shaped curve” that’s symmetrical on both sides."
          },
          { id: 2, 
            title: "Why we should know about Normal distribution?",
            value: "The normal distribution is foundational in machine learning and deep learning. When we built our first model, we use normal distribution to initialize the weight and bias, why we do it? Cus in deep learning, weight and biases in neural networks are often initialized using values drawn from a normal distribution. This helps in ensuring that the model starts training with a balanced approach, preventing issues like vanishing or exploding gradients. No more explaining, we’re gonna cove the more details in Gradient Descent."
          },
          { id: 3, 
            title: "Normal Distribution in Machine Learning",
            value: "In machine learning, linear regression both a foundational algorithm and a gateway to more complex models. It is often one of the first techniques taught because it's straightforward and easy to interpret, making it an essential tool for understanding the basics of predictive modeling. In machine learning, linear regression is used to predict continuous outcomes, such as prices, temperatures, or scores, based on input data."
          },
        ]
      },
      { id: 1,
        name: "2.1 Probability Density Function (PDF)", 
        value: "Let’s dive into the Probability Density Function, or PDF, which is key when talking about the Standard Normal Distribution. No need to stress over memorizing the complicated formula; what’s crucial is grasping the shape of the function and the key parameters that shape it. These parameters dictate whether the bell curve is narrow or wide, giving you a clear picture of how the data is spread out.",
        image: "src/assets/chapter_three/ndistrib_pdf.jpeg",
        content: [
          { id: 1, 
            title: "Bell-shaped:",
            value: "Picture a bell. That’s the shape we’re talking about."
          },
          { id: 2, 
            title: "Mean (µ = 0):",
            value: "This is the peak, or the highest point of the bell. In the standard normal distribution, µ equals 0."
          },
          { id: 3, 
            title: "Standard Deviation (σ = 1):",
            value: "This determines how wide or narrow the bell is. For the standard normal distribution, σ is 1."
          },
          { id: 4, 
            title: "x-axis",
            value: "Represents the possible values of the random variable."
          },
          { id: 5, 
            title: "y-axis:",
            value: "Shows the probability density, or how likely each value on the x-axis is."
          },
        ]
      },
      { id: 2,
        name: "2.2 Cumulative Distribution Function (CDF)", 
        value: "The CDF is visualized as the shaded area under the curve to the left of a specific point, which we’ll call x. It represents the probability that the random variable X will take on a value less than or equal to x.",
        image: "src/assets/chapter_three/ndistrib_cdf.jpeg",
        content: [
          { id: 1, 
            title: "S-shaped:",
            value: "Think of it as an S-curve, but still related to that bell shape."
          },
          { id: 2, 
            title: "A special point - (µ, 0.5):",
            value: "In a standard normal distribution, the CDF at x = 0 is 0.5. This means there’s a 50% chance the variable falls to the left of the mean."
          },
          { id: 3, 
            title: "x-axis",
            value: "Represents the values of the random variable."
          },
          { id: 4, 
            title: "y-axis:",
            value: "Represents the probability density."
          },
        ]
      },
    ]
  },
  { id: 2,
    name: "Loss Function",
    sections: [
      { id: 0,
        name: "3. Loss Function",
        value: "The Loss Function is used to evaluate how well your model’s predictions are performing, the lower the value, the better the model is doing.",
        image: "",
        content: [
          { id: 1,
            title: "What is loss function",
            value: "A Loss Function is essential in evaluating how well your model’s predictions match the actual data. It measures the difference between your model’s predictions and the true values, with the goal being to minimize this difference. In simple terms, a lower loss means your model’s predictions are more accurate."
          },
          { id: 2,
            title: "How it works",
            value: "The loss function calculates the difference between the values predicted by your model and the actual values from your dataset. This difference is then summarized into a single scalar value, representing the “loss” or error."
          },
          { id: 3,
            title: "Why it matters",
            value: "The loss function plays a crucial role in guiding the optimization process. During training, the model uses the loss value to fine-tune its internal parameters (like weights and biases) to reduce the loss, leading to better predictions over time."
          },
          { id: 4,
            title: "Lower is Better",
            value: "The key goal during training is to minimize the loss function. A lower loss indicates that your model’s predictions are getting closer to the actual values, meaning the model is effectively learning."
          }
        ]
      },
      { id: 1,
        name: "3.1 Mean Squared Error (MSE)",
        value: "Mean Squared Error, or MSE for short, is a type of loss function. It’s quite sensitive to outliers because larger errors are squared, making them more significant.",
        image: "",
        content: [
          { id: 1,
            title: "",
            value: "It calculates the average of the squared differences between predicted and actual values. Since larger errors are squared, MSE is more sensitive to outliers, making them more significant."
          },
        ]
      },
      { id: 2,
        name: ":: MSE in straight-line model (part.1)",
        value: "",
        image: "src/assets/chapter_three/mse_one.jpeg",
        content: [
          { id: 1,
            title: "Graph Explanation:",
            value: "This graph displays two lines: the current line (in red), which represents the model’s predictions, and the target line (in blue), which represents the true values we want the model to predict. Data points for different values of x are plotted as dots on both lines—the blue dots correspond to the target values, while the red dots show the predictions made by the current model."
          },
          { id: 2,
            title: "Mathematical Representation:",
            value: "The current line is represented by the equation y = 0.3x + b, where b is the bias term. The target line follows the equation Y = 0.6x + 0.1. The difference between the predictions and the actual target values is illustrated by vertical arrows, showing the gap between the two."
          },
        ]
      },
      { id: 3,
        name: ":: MSE in straight-line model (part.2)",
        value: "",
        image: "src/assets/chapter_three/mse_two.jpeg",
        content: [
          { id: 1,
            title: "MSE Calculation:",
            value: "The Mean Squared Error (MSE) is calculated by averaging the squared differences between the predicted values (from the red line) and the actual target values (from the blue line). This formula measures how far off the predictions are from the actual values."
          },
          { id: 2, 
            title: "Graph of MSE:",
            value: "The graph on the right plots the MSE as a function of the bias term b. The curve shows how the MSE changes as b varies, with a minimum point indicating the optimal value of b that minimizes the error."
          },
          { id: 3, 
            title: "Objective:",
            value: "The goal of training the model is to adjust the bias b (starting simple by considering just one variable) to minimize the MSE. This is visually represented by finding the lowest point on the MSE curve."
          },
        ]
      },
      { id: 4,
        name: "3.2 Loss Curve",
        value: "The Loss Curve is a graph that tracks a model’s error or losses over the course of training.",
        image: "",
        content: [
          { id: 1,
            title: "",
            value: "The Loss Curve is a graphic tool that tracks a model’s error or losses over the training period. It helps us easily understand how well a model is performing during both training and evaluation."
          },
          { id: 2,
            title: "",
            value: "Loss curves visually represent the change in loss over epochs, which are the training iterations. Typically, you’ll see two curves on the same graph: the Training Loss Curve shows how the loss decreases on the training data over epochs, while the Testing Loss Curve—also known as the Validation Loss Curve—illustrates how the loss changes on the testing data over the same period."
          },
        ]
      },
      { id: 5,
        name: ":: Three kinds of Loss Curves",
        value: "",
        image: "src/assets/chapter_three/losscurves.jpg",
        content: [
          { id: 1,
            title: "Underfitting: Falling short",
            value: "In the first graph, labeled “Underfitting,” both the training loss (blue line) and the test loss (red line) are high and decrease slowly. The gap between the losses is small, but both could be lower. This indicates that the model is struggling to learn the underlying patterns in the data."
          },
          { id: 2,
            title: "Overfitting: Too Much of a Good Thing",
            value: "The second graph, labeled “Overfitting,” shows a scenario where the training loss (blue line) decreases rapidly and is much lower than the test loss (red line), which starts to increase after an initial drop. This means the model is performing very well on the training data but poorly on the test data, indicating it has become too specialized to the training set and isn’t generalizing well."
          },
          { id: 3,
            title: "Just Right: Striking the Balance",
            value: "In the third graph, labeled “Just Right,” both the training loss (blue line) and test loss (red line) decrease smoothly and converge at similar low values. This indicates that the model is well-tuned, learning effectively from the training data while also generalizing well to the test data."
          }
        ]
      },
    ]
  },
  { id: 3,
    name: "Gradient Descent",
    sections: [
      { id: 0,
        name: "4. Gradient Descent",
        value: "Gradient Descent is an optimization algorithm that calculates the gradient (slope) using all the samples in the training set to update the model’s parameters during each iteration.",
        image: "",
        content: [
          { id: 1,
            title: "",
            value: "Think of yourself on a mountain, aiming to reach the lowest point, which is the global minimum. The loss function represents the landscape, and the gradients are the slopes that guide you on which direction to step to go downhill."
          },
          { id: 2,
            title: "",
            value: "The gradient is the derivative of the loss function with respect to the model’s parameters, indicating both the direction and the rate at which the loss function is changing."
          },
        ]
      },
      { id: 1,
        name: "4.1 GD SIMULATION (1 parameter)",
        value: "",
        image: "src/assets/chapter_three/gd_one.jpeg",
        content: [
          { id: 0,
            title: "",
            value: "This graph visually demonstrates how Gradient Descent iteratively adjusts the model’s parameters to minimize the error between the model’s predictions (red points) and the actual values (blue points). With each step, the model’s prediction line shifts from  y = 0.3x  towards a line closer to  Y = 0.6x + 0.1 , effectively reducing the overall error. The gradual alignment of the model’s line with the target line showcases the effectiveness of Gradient Descent in enhancing the model’s performance."
          },
          { id: 1,
            title: "Initial Line (Red Line):",
            value: "The red points correspond to the initial predictions made by the model with the initial parameter  b = 0 . The red line represents the model’s starting point before any Gradient Descent iterations have occurred."
          },
          { id: 2,
            title: "Target Line (Blue Line):",
            value: "The blue points represent the ideal target values that the model should predict. The blue line, which passes through these points, is the optimal line that the model aims to approximate through Gradient Descent."
          },
          { id: 3,
            title: "Gradient Descent Iterations:",
            value: "The grey lines between the red and blue lines illustrate how the model’s prediction line shifts after each Gradient Descent iteration. Each update reduces the discrepancy between the model’s predictions and the target values, gradually moving the prediction line closer to the target line. This process demonstrates the model’s convergence towards more accurate predictions."
          },
        ]
      },
      { id: 2,
        name: ":: GD SIMULATION Table",
        value: "",
        image: "src/assets/chapter_three/gd_one_table.jpeg",
        content: [
          { id: 1,
            title: "",
            value: "In this table, we display the parameter  b  across several iterations, along with the corresponding Mean Squared Error (MSE) and the derivative of the MSE."
          },
          { id: 2,
            title: "",
            value: "The algorithm begins with an initial guess for  b  (set to 0 for demonstration purposes) and iteratively updates  b  based on the gradient (slope) of the MSE at that particular point."
          },
          { id: 3,
            title: "",
            value: "The update rule is given by the function, b(n+1) =  b(n) - lr*MSE’(b), and lr is the learning rate, we set it to 0.2."
          },
        ]
      },
      { id: 3,
        name: ":: GD SIMULATION on MSE Curve",
        value: "",
        image: "src/assets/chapter_three/gd_one_mse.jpeg",
        content: [
          { id: 1,
            title: "MSE curve",
            value: "The middle graph shows the MSE curve as a function of  b . The curve represents how the error changes as we adjust  b . The red points on the curve show the progression of the gradient descent, starting from an initial b = 0 and moving towards a lower MSE value as b is updated in each step. The lines touching the curve represent the gradient at each point, guiding the direction and magnitude of the updates."
          },
          { id: 2,
            title: "Derivative of MSE",
            value: "The right graph represents the derivative of the MSE with respect to b. The red dots indicate the values of the gradient at each iteration. The gradient decreases as the algorithm approaches the minimum. The gradient becomes closer to zero as b approaches the value that minimizes the MSE. When the gradient is zero, the algorithm has reached the optimal value of b ."
          },
        ]
      },
      { id: 4,
        name: "4.2 GD SIMULATION (2 parameters)",
        value: "Let’s visualize the 2 parameters Gradient Descent. This 3D surface plot visually represents the process of Gradient Descent in a multi-parameter scenario, where two parameters θ1 and θ2 are being optimized to minimize a loss function, denoted as  Loss(θ1, θ2).",
        image: "src/assets/gradient_descent.jpeg",
        content: [
          { id: 1,
            title: "Surface Plot:",
            value: "The plot shows the loss function’s landscape, with the height of the surface representing the value of the loss function for different values of θ1 and θ2. The goal of Gradient Descent is to navigate this surface to find the point where the loss is minimized, which corresponds to the lowest point on the surface (the global minimum)."
          },
          { id: 2,
            title: "Arrows Indicating Descent:",
            value: "The arrows on the surface indicate the path taken by the Gradient Descent algorithm as it iteratively adjusts θ1 and θ2 to move towards the minimum loss. These arrows demonstrate how the algorithm follows the slope (gradient) of the surface downwards, decreasing the loss with each step."
          },
          { id: 3,
            title: "Contour of the Loss Function:",
            value: "The plot also reflects the contours of the loss function, with the dark regions representing areas of higher loss and the lighter regions indicating lower loss. The gradient arrows move from darker to lighter regions, which visually represents the process of minimizing the loss."
          },
          { id: 4,
            title: "Axes:",
            value: "The x and y axes represent the values of the parameters θ1 and θ2. The z-axis represents the loss value Loss(θ1, θ2)."
          },
        ]
      },
      { id: 5,
        name: ":: GD SIMULATION Table",
        value: "",
        image: "src/assets/chapter_three/gd_two_table.jpeg",
        content: [
          { id: 1,
            title: "",
            value: "Here in this table we display two parameters bias (b) and weights (w) across several iterations, along with the corresponding MSE and the deriveative of MSE. We initialize these two parameters (set b to 0 and w to 0.3)."
          },
          { id: 2,
            title: "",
            value: " Then update w and b based on the gradient of the MSE, and the update rule is pretty much similar to the 1 parameter function. Here's the tricky part, when we're calculating ths slope for one parameter, we treat the other parameter as a constant number."
          },
        ]
      },
      { id: 6,
        name: ":: GD SIMULATION Visualization",
        value: "",
        image: "src/assets/chapter_three/gd_two.jpeg",
        content: [
          { id: 1,
            title: "",
            value: "This graph is a visualization of how Gradient Descent optimizes a linear model by iteratively updating its two parameters: the weight (w) and the bias (b)."
          },
          { id: 2,
            title: "",
            value: "Previously, we focused on just one parameter, the bias ( b ), while keeping the weight ( w = 0.3 ) fixed. This allowed us to understand how Gradient Descent adjusts the bias to minimize error. Now, let’s take it a step further by considering both parameters — weight and bias — simultaneously."
          },
          { id: 3,
            title: "",
            value: "By incorporating both parameters, the Gradient Descent algorithm updates them at each iteration to reduce the model’s error, just as we did with a single parameter. The process involves calculating new values for  w  and  b  based on their gradients and adjusting the model accordingly."
          },
          { id: 4,
            title: "",
            value: "This visualization will help you see how the model evolves when both the weight and bias are adjusted together, showcasing the path Gradient Descent takes to find the optimal parameter values. It’s essentially the same concept as before, just applied to a more complex scenario with two variables."
          },
        ]
      },
    ]
  },
  { id: 4,
    name: "Stochastic Gradient Descent",
    sections: [
      { id: 0,
        name: "5. Stochastic Gradient Descent (SGD)",
        value: "Stochastic Gradient Descent uses one sample or a subset of the samples randomly to do the same thing as regular Gradient Descent.",
        image: "",
        content: [
          { id: 1,
            title: "What is SGD?",
            value: "Stochastic Gradient Descent uses one or a subset of the samples randomly to do the same thing as regular Gradient Descent. Stochastic Gradient Descent (SGD) is an optimization technique used in machine learning and deep learning. Unlike the regular Gradient Descent, which uses the entire dataset to compute the gradient and update the parameters, SGD uses only one sample or a small, randomly selected subset of samples (also called a mini-batch) at each iteration to compute the gradient."
          },
          { id: 2,
            title: "Why SGD?",
            value: "By using only one data or a small subset of data to update the parameters, SGD can be much faster and more memory-efficient than regular Gradient Descent, especially when dealing with large datasets. The more frequent parameter updates because of the faster computation, and frequent parameters’ updating can lead to faster convergence, though the path to convergence might be noisier and less smooth compared to regular Gradient Descent."
          },
        ]
      },
      { id: 1,
        name: ":: SGD vs. GD",
        value: "Stochastic Gradient Descent uses one sample or a subset of the samples randomly to do the same thing as regular Gradient Descent.",
        image: "src/assets/chapter_three/sgd_one.jpeg",
        content: [
          { id: 1,
            title: "Large dataset",
            value: "GD: Struggles with large datasets because it requires processing the entire dataset for each update. SGD: Handles large datasets efficiently by using only one or a small subset of samples per update."
          },
          { id: 2,
            title: "Lots of parameters:",
            value: "GD: Handles well but can be slow due to the need to compute gradients for all parameters at once. SGD: Scales better with a large number of parameters due to more frequent updates."
          },
          { id: 3,
            title: "Parameter Updating:",
            value: "GD: Updates parameters after computing gradients from the whole dataset, leading to stable and accurate updates. SGD: Updates parameters more frequently with each sample or subset, which introduces noise but speeds up the process."
          },
          { id: 4,
            title: "RAM usage:",
            value: "GD: Requires more RAM as it processes the entire dataset at each step. SGD: Requires less RAM since it processes only a small subset of the data at a time."
          },
          { id: 5,
            title: "Convergence Rate:",
            value: "GD: Slower but more stable convergence to a precise minimum. SGD: Faster convergence but with more variability and potential oscillation around the minimum."
          },
          { id: 6,
            title: "Accuracy:",
            value: "GD: Often results in higher final accuracy, as updates are less noisy and reflect the overall trend in the data. SGD: May result in slightly lower accuracy due to noisy updates, but can be improved with techniques like learning rate decay and momentum."
          },
        ]
      },
      { id: 2,
        name: ":: SGD vs. GD",
        value: "The graph effectively illustrates the key differences between Gradient Descent (GD) and Stochastic Gradient Descent (SGD) in the context of how each algorithm updates parameters to minimize a loss function.",
        image: "src/assets/chapter_three/sgd_two.jpeg",
        content: [
          { id: 1,
            title: "Gradient Descent:",
            value: "Provides a smooth, stable, and consistent path to the minimum but is computationally expensive and slower for large datasets."
          },
          { id: 2,
            title: "Stochastic Gradient Descent:",
            value: "Offers a faster, more memory-efficient alternative but introduces noise in the updates, leading to a less predictable and more erratic path towards the minimum. This noise can be beneficial for escaping local minima but may require techniques like learning rate decay to stabilize the final convergence."
          },
        ]
      },
    ]
  },
  { id: 5,
    name: "",
    sections: [
      { id: 0,
        name: "6. Learning Rate (lr)",
        value: "Learning Rate is a parameter in an optimization algorithm that determines the step size at each iteration.",
        image: "",
        content: [
          { id: 1,
            title: "",
            value: "Learning Rate is a parameter in an optimization algorithm that determines the step size at each iteration."
          },
          { id: 2,
            title: "",
            value: "The learning rate is indeed a critical parameter in optimization algorithms like Gradient Descent and its variants. It controls the size of the steps taken towards the minimum of the loss function during the training process of a machine learning model."
          },
          { id: 3,
            title: "",
            value: "The learning rate is a fundamental aspect of training machine learning models. It directly influences the training process’s efficiency, speed, and stability. By understanding and tuning the learning rate appropriately, you can significantly improve the performance and convergence of your models, ensuring they learn effectively from the data."
          },
          { id: 4,
            title: "",
            value: "The learning rate is a critical hyperparameter that directly influences how the training loss behaves during model optimization. An optimal learning rate results in efficient and stable convergence of the training loss. In contrast, an excessively high or low learning rate can lead to oscillations, divergence, or slow convergence, making it crucial to carefully tune this parameter for effective training."
          },
        ]
      },
      { id: 1,
        name: ":: Learing Rate curves",
        value: "This graph illustrates the relationship between the learning rate and the training loss. Here’s a summary:",
        image: "src/assets/chapter_three/lr.jpeg",
        content: [
          { id: 1,
            title: "Flat Region:",
            value: "At very low learning rates, the training loss remains relatively constant, shown by the flat region at the beginning of the curve. The model is updating its parameters too slowly to make significant progress, leading to minimal changes in the loss."
          },
          { id: 2,
            title: "Descent Region:",
            value: "As the learning rate increases, the training loss starts to decrease, which is indicated by the descending part of the curve. This is where the learning rate is effective in guiding the model toward the minimum loss. The model is making more significant updates, improving its performance."
          },
          { id: 3,
            title: "Optimal Point:",
            value: "There is a point, marked by a star, where the learning rate results in the lowest training loss. This represents the optimal learning rate, where the balance between convergence speed and stability is ideal."
          },
          { id: 4,
            title: "Explosion Region:",
            value: "Beyond the optimal point, as the learning rate continues to increase, the training loss suddenly rises sharply. This is the explosion region where the learning rate is too high, causing the model to overshoot the minimum, leading to divergence or even instability in the training process."
          },
        ]
      },
      { id: 2,
        name: ":: Too Low (lr=0.05)",
        value: "",
        image: "src/assets/chapter_three/lr_low.jpeg",
        content: [
          { id: 1,
            title: "",
            value: "The learning rate is very small, leading to tiny steps during each iteration of Gradient Descent. The model converges towards the minimum, but it does so very slowly. The training process becomes lengthy, requiring many more iterations to achieve a minimal loss. The gradient steps are small and tightly packed, showing slow progress toward the optimal point."
          },
        ]
      },
      { id: 3,
        name: ":: Too High (lr=0.8)",
        value: "",
        image: "src/assets/chapter_three/lr_low.jpeg",
        content: [
          { id: 1,
            title: "",
            value: "The learning rate is too large, causing the model to take large steps during each iteration. The model oscillates around the minimum or even diverges, leading to instability and failure to converge to the optimal solution. The gradient steps are erratic, often overshooting the optimal point and resulting in an increase in loss rather than a decrease."
          },
        ]
      },
      { id: 4,
        name: ":: Just Right (lr=0.2)",
        value: "",
        image: "src/assets/chapter_three/lr_low.jpeg",
        content: [
          { id: 1,
            title: "",
            value: "The learning rate is appropriately set, allowing the model to make significant progress without overshooting the minimum. The model converges quickly and efficiently to the minimum loss. This represents an optimal learning rate where the balance between speed and accuracy is achieved. The gradient steps are well-paced, showing a steady and effective progression toward the optimal point."
          },
        ]
      },
    ]
  }
]