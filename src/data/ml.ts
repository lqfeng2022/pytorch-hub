import mlcomparing from '../assets/mlcomparing.jpeg'
import svms from '../assets/svms.jpeg'
import ann from '../assets/ann.jpeg'
import decisionTree from '../assets/decisionTree.jpeg'

export default [
  { id: 1, 
    name: "Machine Learning", 
    image: null,
    value: "Machine Learning (ML) is a subset of AI that perform Specific Tasks without being Explicitly Programmed.",
    content: [
      { id: 1, 
        title: "",
        value: "Here's the definition of Machine Learning, and let's break it down with an example."
      },
      { id: 2, 
        title: "",
        value: "Imagine I ask you to write a program that scans an image to determine whether it shows a cat or a dog. If you tried to build this program using traditional programming techniques, it would quickly become overly complicated. You'd need to create a ton of rules to account for specific curves, edges, colors, ear shapes, and other features to distinguish between a cat and a dog. And if I handed you a black-and-white photo, those rules might not work, so you'd have to rewrite them. Or if the image was taken from a different angle, your program might struggle to identify it correctly. So, trying to solve this problem with traditional programming would be not only complex but potentially impossible in some cases."
      },
      { id: 3,
        title: "",
        value: "Now, imagine I asked you to extend this program to recognize three kinds of animals—cats, dogs, and horses. Once again, you'd have to rewrite all those rules. That's just not practical. This is where Machine Learning comes in."
      },
      { id: 4,
        title: "",
        value: "ML is specifically designed to handle challenges like this. Here's how it works: We build a model and feed it a lot of data, like thousands or even tens of thousands of pictures of cats and dogs. The model then learns patterns from this input data. After training, we can show the model a new picture of a cat or a dog that it hasn't seen before and ask it whether it's a cat or a dog. The model will then give us an answer with a certain level of accuracy. The more data we provide, the more accurate the model becomes."
      }
    ]
  },
  { id: 2,
    name: "2.1 Machine Learing V.S. Traditinal Programming",
    image: mlcomparing,
    value: "",
    content: [
      { id: 0, 
        title: "1) Traditional Programming", 
        value: "Traditional programming relies on rule-based coding, where you explicitly define the rules for the computer to follow. In this approach, you determine the logic and sequence of operations, and the computer simply executes those instructions."
      },
      { id: 1, 
        title: "2) Machine Learning",
        value: "Machine Learning, on the other hand, is a data-driven approach. Instead of writing explicit rules, you provide a model and a large dataset. The model is then trained to learn patterns and relationships between the input and output on its own."
      },
    ]
  },
  { id: 3, 
    name: "1) Support-vector machines (SVMs)", 
    image: svms,
    value: "",
    content: [
      { id: 0, 
        title: "",
        value: "SVMs, or Support Vector Machines, a type of supervised learning method that used for both classification and regression tasks. When you have a set of training examples, each labeled as belonging to one of two categories, an SVM algorithm builds a model to predict which category a new example will fall into."
      },
      { id: 1, 
        title: "",
        value: "Not only the linear regression, SVMs can also handle non-linear classification."
      }
  ]},
  { id: 4, 
    name: "2) Decision Tree",
    image: decisionTree,
    value: "",
    content: [
      { id: 0, 
        title: "",
        value: "It is a hierarchical model that uses a tree-like structure to represent decisions and their possible consequences. It's more like a human-logic simulation than the biological mimics of ANNs.A decision tree is like a chat flow, where each internal note represents a TEST on an attribute, each branch represents the output of the test, and each leaf node represents a class label. The paths from the root to the leaves show the classification rules."
      },
  ]},
  { id: 5, 
    name: "3) Artificial Neural Networks (ANNs)", 
    image: ann,
    value: "",
    content: [
      { id: 0, 
        title: "",
        value: "It's a connectionist system inspired by the animal brain - especially the human brain. The brain is basically a network of neurons that transmit electrical signals."
      },
      { id: 1, 
        title: "",
        value: "An Artificial Neural Network (ANN) mimics this by using a collection of connected units or 'Artificial Neurons', similar to how brain neurons work. Each artificial neuron in this model receive signals, processes them, and then sends them on to the other neurons."
      }
  ]},
]