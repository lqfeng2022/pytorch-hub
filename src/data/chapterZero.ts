import { 
  ai_why, 
  ai_turningTest,
  ai_expertSystem,
  ai_connectionism,
  ai_convolutinal,
  ai_alphgo,
  ai_openai,
  ml_vs_traditonal,
  ml_svms,
  ml_decisionTree,
  ml_anns,
  dl_vs_ml_ai,
  dl_cat_ml,
  dl_cat_dl,
  dl_vs_ml,
  dl_apps,
  nn_model,
  nn_neurons,
  pt_features,
  pt_trends,
  pt_companies,
  how_take_course
 } from '../assets/chapter_zero'


export default [
  { id: 1, 
    name: "Artificial Intelligence",
    sections: [
      { id: 1,
        name: "1. Artificial Intelligence", 
        value: "Artificial Intelligence (AI) is a branch of Computer Science that makes Machines and Systems perform tasks that Simulate Human Intelligence.",
        image: "",
        content: [
          { id: 1, 
            title: "",
            value: "Artificial Intelligence (AI) is a branch of computer science dedicated to developing machines and systems that can perform tasks traditionally requiring human intelligence. These tasks include learning from experience, adapting to new information, reasoning through complex problems, recognizing patterns, understanding and processing natural language, and even making decisions."
          },
          { id: 2, 
            title: "",
            value: "AI systems can be categorized into different levels, from narrow AI, which is designed for specific tasks like facial recognition or language translation, to general AI, which would have the capability to perform any intellectual task that a human can do. Although we are currently at the stage of narrow AI, advancements in the field are rapidly evolving, pushing the boundaries of what machines can achieve."
          },
          { id: 3,
            title: "",
            value: "AI technologies power many aspects of our daily lives, from recommendation algorithms on streaming services to virtual assistants like Siri and Alexa, autonomous vehicles, and sophisticated data analysis tools used in industries like healthcare, finance, and beyond. As AI continues to evolve, it holds the potential to transform industries, drive innovation, and address complex global challenges, but it also raises important ethical and societal considerations that need to be carefully managed."
          }
        ]
      },
      { id: 2, 
        name: "1.1 Why we should know about AI?",
        value: "",
        image: ai_why,
        content: [
          { id: 1, 
            title: "ChatGPT", 
            value: "When ChatGPT-3.5 came out on November 30, 2022, it was the first time that AI impressed us by its human-like text. If you're interested in ChatGPT and artificial intelligence, you should take this course. I am gonna lift the veil of mystery - ChatGPT, the most successful and powerful artificial intelligence product."
          },
          { id: 2, 
            title: "AI is everywhere",
            value: "Is all around us. Even now, as you're watching this video. You can see it when you look at the camera or when you pick up your iPhone. You can feel it when you go to work,  when you go to restaurant, when you go to shop, when you chat with your friends. It's the world that has been pulled over your eyes, to blind you from the truth. The truth is that the AI is watching you! It knows where you go, what you eat, what you bought, and even what you talked with your friends.."
          },
          { id: 3,
            title: "BOOST your productivity at work",
            value: "AI is awesome, especially when we wanna handle the repetitive tasks like data entry, managing schedules, summarizing information, and even analyzing papers. With AI, taking care of those boring tasks, we can focus more on higher-value tasks, like strategy analysis, employee engagement."
          },
          { id: 4,
            title: "AI powered Robot",
            value: "What if you have an AI robot that can cook dinner, wash the dishes, clean the floor, take care of your parents, and even chat with you like an old friend. Would you want to pay for something like that?"
          }
        ]
      },
      { id: 3,
        name: ":: Turning Test", 
        value: "",
        image: ai_turningTest,
        content: [
          { id: 1, 
            title: "",
            value: "Alan Turing was the first to really explore what he called 'machine intelligence'. In his 1950 paper, 'Computing Machinery and Intelligence', he introduced the Turing test and argued that machines could actually be intelligent. This was a major milestone and laid the foundation for AI research, which officially became an academic field in 1956."
          },
          { id: 2, 
            title: "",
            value: "The typical way to understand the Turing test is this: there are 3 players - A, B, and C. Player C is the interrogator, and his job is to figure out which of the other two, A or B, is a computer and which one is a human. The tricky part? The interrogator can only ask written questions to make that decision."
          }
        ]
      },
      { id: 4, 
        name: ":: Expert System", 
        value: "",
        image: ai_expertSystem,
        content: [
          { id: 1, 
            title: "",
            value: "In the early 1980s, AI research got a boost thanks to the commercial success of expert systems - AI programs - that mimicked the knowledge and analytical skills of human experts."
          },
        ]
      },
      { id: 5, 
        name: ":: Connectionism + Neural Network", 
        value: "",
        image: ai_connectionism,
        content: [
          { id: 1, 
            title: "",
            value: "A few years later, one of the biggest breakthroughs was the revival of CONNECTIONISM, led by Geoffrey Hinton and others, which brought neural network research back into the spotlight - it's all about simulating how the brain's neural networks work."
          },
          { id: 2, 
            title: "",
            value: "The core idea behind connectionism is that mental processes can be understood as networks that interconnected units. These units represent neurons, and the connections between them act like synapses, just like in the human brain."
          }
        ]
      },
      { id: 6, 
        name: ":: Convolutional Neural Network", 
        value: "",
        image: ai_convolutinal,
        content: [
          { id: 1, 
            title: "",
            value: "In 1990, Yann LeCun showed that convolutional neural networks could accurately recognize handwritten digits. This was one of the first major successes for neural networks."
          }
        ]
      },
      { id: 7, 
        name: ":: AlphaGo", 
        value: "",
        image: ai_alphgo,
        content: [
          { id: 1,
            title: "",
            value: "Deep learning started taking over industry benchmarks in 2012, thanks to better hardware and access to huge amounts of data."
          },
          { id: 2,
            title: "",
            value: "Then, in 2015, DeepMind's AlphaGo made headlines by beating the world champion Go player. The program was only taught the rules and figured out its own strategy."
          }
        ]
      },
      { id: 8, 
        name: ":: ChatGPT", 
        value: "",
        image: ai_openai,
        content: [
          { id: 1, 
            title: "",
            value: "Fast forward to the end of 2022, and OpenAI released GPT-3.5, a large language model that can generate text that feels like it was written by a human."
          }
        ]
      },
    ]
  },
  { id: 2, 
    name: "Machine Learning",
    sections: [
      { id: 0, 
        name: "2. Machine Learning", 
        value: "Machine Learning (ML) is a subset of AI that perform Specific Tasks without being Explicitly Programmed.",
        image: "",
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
            value: "Now, imagine I asked you to extend this program to recognize three kinds of animals - cats, dogs, and horses. Once again, you'd have to rewrite all those rules. That's just not practical. This is where Machine Learning comes in."
          },
          { id: 4,
            title: "",
            value: "ML is specifically designed to handle challenges like this. Here's how it works: We build a model and feed it a lot of data, like thousands or even tens of thousands of pictures of cats and dogs. The model then learns patterns from this input data. After training, we can show the model a new picture of a cat or a dog that it hasn't seen before and ask it whether it's a cat or a dog. The model will then give us an answer with a certain level of accuracy. The more data we provide, the more accurate the model becomes."
          }
        ]
      },
      { id: 1,
        name: "2.1 Machine Learning vs. Traditional Programming",
        value: "",
        image: ml_vs_traditonal,
        content: [
          { id: 1, 
            title: "Traditional Programming", 
            value: "Traditional programming relies on rule-based coding, where you explicitly define the rules for the computer to follow. In this approach, you determine the logic and sequence of operations, and the computer simply executes those instructions."
          },
          { id: 2, 
            title: "Machine Learning",
            value: "Machine Learning, on the other hand, is a data-driven approach. Instead of writing explicit rules, you provide a model and a large dataset. The model is then trained to learn patterns and relationships between the input and output on its own."
          },
        ]
      },
      { id: 2, 
        name: ":: Support-vector machines (SVMs)", 
        value: "",
        image: ml_svms,
        content: [
          { id: 1, 
            title: "",
            value: "SVMs, or Support Vector Machines, a type of supervised learning method that used for both classification and regression tasks. When you have a set of training examples, each labeled as belonging to one of two categories, an SVM algorithm builds a model to predict which category a new example will fall into."
          },
          { id: 2, 
            title: "",
            value: "Not only the linear regression, SVMs can also handle non-linear classification."
          }
      ]},
      { id: 3, 
        name: ":: Decision Tree",
        value: "",
        image: ml_decisionTree,
        content: [
          { id: 1, 
            title: "",
            value: "It is a hierarchical model that uses a tree-like structure to represent decisions and their possible consequences. It's more like a human-logic simulation than the biological mimics of ANNs.A decision tree is like a chat flow, where each internal note represents a TEST on an attribute, each branch represents the output of the test, and each leaf node represents a class label. The paths from the root to the leaves show the classification rules."
          },
      ]},
      { id: 4, 
        name: ":: Artificial Neural Networks (ANNs)", 
        value: "",
        image: ml_anns,
        content: [
          { id: 1, 
            title: "",
            value: "It's a connectionist system inspired by the animal brain - especially the human brain. The brain is basically a network of neurons that transmit electrical signals."
          },
          { id: 2, 
            title: "",
            value: "An Artificial Neural Network (ANN) mimics this by using a collection of connected units or 'Artificial Neurons', similar to how brain neurons work. Each artificial neuron in this model receive signals, processes them, and then sends them on to the other neurons."
          }
      ]},
    ]
  },
  { id: 3, 
    name: "Deep Learning",
    sections: [
      { id: 0, 
        name: "3. Deep Learning", 
        value: "Deep Learning (DL) is a subset of Machine Learning that utilizes neural networks with multiple layers to model complex patterns in data.",
        image: "",
        content: [
          { id: 1, 
            title: "",
            value: "Let's use the same cat-and-dog example to explain what Deep Learning is."
          },
          { id: 2, 
            title: "",
            value: "In traditional Machine Learning, we rely on a technique called feature extraction. Here's how it works: First, we manually extract features from the images, like edges, textures, colors, or ear shapes that are commonly found in cats and dogs. Then, we feed these features into a Machine Learning model, such as a Decision Tree. The model learns patterns from these features during training. When given a new image, the model uses the learned patterns to classify whether it's a cat or a dog."
          },
          { id: 3,
            title: "",
            value: "The critical aspect of traditional Machine Learning is that we have to manually select and extract the features that will help the model make accurate classifications."
          },
          { id: 4,
            title: "",
            value: "But what if we used a Deep Learning model to determine whether an image is of a cat or a dog? First and foremost, we don't need to manually pick out the features. Instead, we feed the raw images directly into a neural network. A Convolutional Neural Network (CNN), for example, will automatically learn the features during the training process. In the initial layers, it might detect basic patterns like edges. As the data passes through more layers, the network learns to recognize more complex features like ear shapes, eventually distinguishing between cats and dogs on its own."
          },
          { id: 5,
            title: "",
            value: "Once trained, the CNN can take a new image and determine whether it's a cat or a dog based on the features it has learned."
          },
          { id: 6,
            title: "",
            value: "The key point is that Deep Learning automates the feature extraction process, allowing the model to learn directly from raw data. This often results in better performance on complex tasks like image recognition, compared to traditional Machine Learning methods."
          },
        ]
      },
      { id: 1, 
        name: "3.1 Deep Learning < Machine Learning < AI",
        value: "In summary,  AI is the overall goal of creating intelligent systems. ML is a way to achieve AI by enabling machines to learn from data. DL is a more advanced technique within ML that uses layered neural networks to learn from large amounts of data.",
        image: dl_vs_ml_ai,
        content: [
          { id: 1, 
            title: "", 
            value: "Artificial Intelligence (AI) is the broadest concept, referring to the development of computer systems that can perform tasks typically requiring human intelligence."
          },
          { id: 2, 
            title: "",
            value: "Machine Learning (ML) is a subset of AI. It focuses on creating algorithms and models that allow computers to learn from and make predictions or decisions based on data."
          },
          { id: 3, 
            title: "",
            value: "Deep Learning (DL) is a subset of Machine Learning. It involves neural networks with many layers (hence DEEP) that can automatically learn complex patterns in data."
          },
        ]
      },
      { id: 2, 
        name: ":: Machine Learning Model (Decision Tree)",
        value: "",
        image: dl_cat_ml,
        content: [
          { id: 1, 
            title: "",
            value: "In this process, the input data must be structured, meaning we need to manually extract features from the image, or we can use another algorithms to automate this task.s"
          },
          { id: 2, 
            title: "",
            value: "The decision tree model then comes into play by learning to split the data based on these input features. It creates branches according to certain conditions (e.g., “if the image has a certain texture, go left; otherwise, go right”). The tree keeps splitting the data until it reaches a decision point, ultimately classifying the image as either a “cat” or “not a cat”."
          },
          { id: 3, 
            title: "",
            value: "When presented with a new image, the same features must be extracted and passed through the decision tree. While this process is straightforward, it can sometimes feel a bit repetitive."
          },
        ]
      },
      { id: 3, 
        name: ":: Deep Learning Model (Neural Network)",
        value: "",
        image: dl_cat_dl,
        content: [
          { id: 1, 
            title: "",
            value: "Here, the input data is a raw image, which is unstructured and doesn't need to be converted into structured data like in traditional machine learning. The neural network automatically extracts features, such as edges, textures, and patterns from the image in the initial layers. These features are then combined in deeper layers to detect more complex structures like fur patterns, eyes, and ears."
          },
          { id: 2, 
            title: "",
            value: "The neural network learns through backPropagation, adjusting its parameters based on the errors it makes during training on a labeled dataset (images labeled as “cat” or “not a cat”). The model gets better over time at identifying the features that distinguish a cat from other objects."
          },
          { id: 3, 
            title: "",
            value: "Once trained, the neural network can process a new image through its layers and output a probability score, indicating the likelihood that the image contains a cat. Based on this score, the model can classify the image as “cat” or “not cat”."
          },
        ]
      },
      { id: 4, 
        name: "3.3 Differences between Machine Learnig and Deep Learning",
        value: "",
        image: dl_vs_ml,
        content: [
          { id: 1, 
            title: "Structure Data",
            value: "Structured data is organized in a clear, predictable format, usually in tables with rows and columns, like a spreadsheet or a database. Each piece of data fits into a specific category, making it easy to search, analyze, and manage. Like numbers, dates, strings in Spreadsheets and Databases."
          },
          { id: 2, 
            title: "Unstructured Data",
            value: "Unstructured data is information that doesn't have a pre-defined or organized format. Unlike structured data, it doesn't fit neatly into tables, rows, or columns, making it harder to store, search, and analyze with traditional tools, like images, videos, audio recordings, text documents, webpages and so on."
          },
          { id: 3, 
            title: "Dataset",
            value: "Dataset is a collection of data, often organized in a structured format, that is used for analysis, research, or training machine learning models. Deep learning needs vast amounts  of data because they learn complex patterns, require lots of examples to generalize well, and often work with high-dimensional data. The more data the model has, the better it can perform, making large datasets essential for successful deep learning applications."
          },
          { id: 4, 
            title: "Algorithm",
            value: "Neural networks have become the most powerful models in the field of machine learning."
          },
          { id: 5, 
            title: "Supervised Learning",
            value: "Supervised learning is a type of machine learning where a model is trained on a labeled dataset. In this context, “labeled” means that each training example is paired with an output label, which represents the correct answer. The goal of supervised learning is to teach the model to make predictions or decisions based on new, unseen data. Supervised learning is all about teaching a model to predict or classify data by learning from examples where the correct answer is already known. It's one of the most common and powerful methods in machine learning, particularly useful when you have a large amount of labeled data to train on."
          },
          { id: 6, 
            title: "Unsupervised Learning",
            value: "Unsupervised learning is a type of machine learning where the model is trained on a dataset without explicit labels or output categories. Unlike supervised learning, the model is not given the correct answers or guidance during training. Instead, it tries to find patterns, relationships, or structures within the data on its own. Unsupervised learning is like exploring unknown territory— the model looks for patterns and relationships within data without being told what to find. It's particularly useful when you have a lot of data but don't know exactly what you're looking for, or when you want to uncover hidden structures or patterns within the data."
          },
        ]
      },
      { id: 5,
        name: "3.4 Deep Learning Applications",
        value: "",
        image: dl_apps,
        content: [
          { id: 1, 
            title: "Image Recognition",
            value: "Image recognition is a part of computer vision that helps computers figure out what's in a picture or camera. Whether it's a photo of your dog, a face, or a stop sign. Image recognition lets the computer look at the image, analyze it, and then tell you what it sees. Image recognition is everywhere, like when you pick up your iPhone, you gotta use it to recognizes your face to unlock, like helping self-driving or autopilot cars see and avoid obstacles on the road."
          },
          { id: 2, 
            title: "Natural Language Processing (NLP)", 
            value: "Natural language processing or NLP that helps computers understand and interact with human language, it's all about making computers smart enough to read, write, and talk in a way that makes sense to us. NLP is used in a lot of cool ways, for example, it can translate languages, like Google Translate does, turn speech to text, like when you talk to Siri, and generate natural language text, such as writing stories and articles, like ChatGPT generating text. In short, NLP is what makes it possible for computers to communicate with us in a natural way, whether that's through text or speech."
          },
          { id: 3, 
            title: "Speech Recognition", 
            value: "Speech recognition is a technology that enables computers to understand and process spoken language. It's what allows your phone, smart speakers, and other devices to “listen” to what you say and response accordingly. Speech recognition is what powers voice assistant like Siri, allowing you to interact with your devices just by talking to them."
          },
          { id: 4, 
            title: "A Recommendation System",
            value: "A recommendation is a type of technology used to suggest products, services, or content to users based on their preferences, behaviors, or other data. It's what helps online platforms like Amazon, JD show you things you might like. In everyday life, recommendation systems are all around you, helping you discover new products, movies, music, or even friends on social media. They're designed to make your experience more personalized and relevant by showing you things that match your tastes and interests. However, it can also come with risks like privacy issues and the potential to limit what we see (the filter bubble). "
          },
        ]
      }
    ]
  },
  { id: 4, 
    name: "Neural Network",
    sections: [
      { id: 0, 
        name: "4. Neural Network", 
        value: "A Neural Network (NN) is a type of algorithm designed to mimic the way the human brain processes information and the connections between neurons.",
        image: "",
        content: [
          { id: 1, 
            title: "",
            value: "Neural networks are the heart of Deep Learning. Because they have a unique ability to learn and make decision in a way that mimics the human brain."
          },
          { id: 2, 
            title: "",
            value: "Neural networks are composed of layers of interconnected nodes, or neurons, that process data in a manner inspired by the structure of the human brain. Each neuron receives input, processes it, and passes it on to the next layer of neurons. As data flows through these layers, the network learns to recognize patterns and make decisions based on the data it has seen."
          },
          { id: 3, 
            title: "",
            value: "What makes neural networks particularly powerful is their ability to automatically adjust the connections between neurons (called weights) during the training process. This adjustment allows the network to improve its accuracy over time, making it capable of handling complex tasks like image and speech recognition, language translation, and even playing games."
          },
          { id: 4, 
            title: "",
            value: "Neural networks can also generalize from the data they've been trained on, meaning they can make accurate predictions or decisions even when faced with new, unseen data. This generalization is what allows them to be so effective in real-world applications, where the exact conditions of the training data are rarely met."
          },
          { id: 5, 
            title: "",
            value: "In summary, neural networks are essential to Deep Learning because they offer a flexible, powerful method for machines to learn from data and make decisions, all while mimicking the processes of the human brain."
          },
        ]
      },
      { id: 1,
        name: "4.1 Neural Network Architecture",
        value: "A neural network consists of layers of interconnected units called neurons or nodes. Here's a breakdown of its basic components:",
        image: nn_model,
        content: [
          { id: 1, 
            title: "Neurons",
            value: "These are the fundamental processing units of the network, similar to neurons in the brain. Each neuron receives input, processes it (typically by applying a weighted sum followed by an activation function), and passes the output to the next layer. The activation function introduces non-linearity, allowing the network to model complex patterns."
          },
          { id: 2, 
            title: "Connections", 
            value: "Neurons in one layer are connected to neurons in the next layer through weights. These connections can be fully connected (where every neuron in one layer connects to every neuron in the next) or sparsely connected. The weights on these connections are the parameters that the network learns during training, adjusting to minimize the error in predictions."
          },
          { id: 3, 
            title: "Input", 
            value: "The first layer of the network, where data is fed into the network. Each neuron in this layer represents a feature of the input data. The number of neurons in this layer typically equals the number of input features."
          },
          { id: 4, 
            title: "Hidden Layers",
            value: "These are the layers between the input and output layers. Hidden layers perform computations and extract features from the input data. The term “deep learning” refers to networks with many hidden layers. The number of neurons and hidden layers can vary depending on the complexity of the task and are often fine-tuned as hyperParameters."
          },
          { id: 5, 
            title: "Output",
            value: "The final layer of the network that provides the output, such as a classification, prediction, or other result based on the processed input. The number of neurons in the output layer corresponds to the number of output classes in classification tasks or to the number of predicted values in regression tasks."
          },
        ]
      },
      { id: 2,
        name: "4.2 What is Neurons in Brain?",
        value: "",
        image: nn_neurons,
        content: [
          { id: 1, 
            title: "Neurons",
            value: "Neurons are the fundamental working units of the brain and the nervous system. These specialized cells transmit information to each other through electrical and chemical signals. A single neuron can process signals from many other neurons and then pass the processed signal along to others. Each neuron consists of three main parts: the cell body (soma), dendrites (which receive signals), and an axon (which sends signals)."
          },
          { id: 2, 
            title: "Neural Network", 
            value: "Neurons are interconnected, forming complex neural networks that power all of the brain's functions, including seeing, hearing, smelling, tasting, feeling, thinking, and even dreaming. These networks are not static; they continuously change and adapt through a process called synaptic plasticity, which is essential for learning and memory."
          },
          { id: 3, 
            title: "THE NERVOUS SYSTEM", 
            value: "The nervous system, with the brain at its core, acts as the body's communication network. It controls everything we do, from breathing and moving to thinking, dreaming, and feeling. The nervous system includes not just the brain but also the spinal cord and peripheral nerves, which allow the brain to communicate with and control the rest of the body. This network manages both voluntary actions (like moving) and involuntary actions (like breathing)."
          },
          { id: 4, 
            title: "NNs (Brain) vs. NNs(Deep Learning)", 
            value: "The structure and function of neural networks in the brain have inspired the development of artificial neural networks in deep learning. While biological neural networks in the brain consist of billions of interconnected neurons that adapt and learn from experiences, deep learning neural networks are composed of artificial neurons arranged in layers. These artificial networks are designed to mimic the way the human brain processes information. In deep learning, neural networks automatically learn features and patterns from large datasets, similar to how the brain learns from sensory input. However, deep learning models, while powerful, are still much simpler and less flexible than the brain's incredibly complex networks."
          },
        ]
      }
    ]
  },
  { id: 5,
    name: "Frameworks and Libraries",
    sections: [
      { id: 0, 
        name: "PyTorch",
        value: "",
        image: "",
        content: [
          { id: 1,
            title: "",
            value: "PyTorch is a popular open-source deep learning framework based on the Torch library, developed by Meta AI. It's incredibly user-friendly, making it a go-to tool for building and training neural networks, especially if you're looking for something intuitive and flexible."
          },
          { id: 2, 
            title: "",
            value: "Widely used in both research and industry, PyTorch is powerful yet accessible. Whether you're a beginner in deep learning or working on advanced AI projects, PyTorch is a great choice. It's also backed by a strong community, so you'll find plenty of tutorials, resources, issue trackers, and support available."
          }
        ],
      },
      { id: 1, 
        name: "TensorFlow",
        value: "",
        image: "",
        content: [
          { id: 1,
            title: "",
            value: "TensorFlow is one of the most popular deep learning frameworks, developed by the Google Brain team. It's more structured than PyTorch, making it ideal for scaling up projects and deploying them in production, especially when you need to run them across multiple devices or in the cloud."
          },
          { id: 2, 
            title: "",
            value: "One of the standout features of TensorFlow is its extensive collection of built-in tools and libraries, allowing you to create anything from simple neural networks to complex, state-of-the-art models. With TensorFlow Lite and TensorFlow.js, you can even run your models on mobile devices or in the browser."
          },
          { id: 3, 
            title: "",
            value: "While TensorFlow can be challenging to learn initially, once you're up and running, it proves to be incredibly versatile. Whether you're just starting out or building advanced models, TensorFlow is a solid choice, backed by a massive community and abundant resources to support your journey."
          }
        ],
      },
      { id: 2, 
        name: "NumPy",
        value: "",
        image: "",
        content: [
          { id: 1,
            title: "",
            value: "NumPy is one of the most widely used scientific computing libraries, designed to perform a variety of mathematical operations on arrays and matrices. Whether you're working with matrices, statistics, or just large sets of numbers, NumPy is an essential tool."
          },
          { id: 2, 
            title: "",
            value: "It serves as the foundation for many other data science libraries, such as Pandas, TensorFlow, and PyTorch. Getting comfortable with NumPy is a crucial step if you're venturing into data science or machine learning. What makes NumPy stand out is its efficiency. It allows you to perform complex calculations on large datasets with remarkable speed. Additionally, it offers a wide range of built-in functions for tasks like linear algebra, random number generation, and much more."
          },
        ],
      },
      { id: 3, 
        name: "MatPLotLib",
        value: "",
        image: "",
        content: [
          { id: 1,
            title: "",
            value: "Matplotlib is a powerful data visualization library used for creating a wide range of plots and graphs. Whether you need a simple line graph, a detailed bar chart, or even a complex plot, Matplotlib has you covered. It's incredibly versatile, allowing you to create almost any type of chart or graph you can imagine."
          },
        ],
      },
      { id: 4, 
        name: "Scikit-Learn",
        value: "",
        image: "",
        content: [
          { id: 1,
            title: "",
            value: "Scikit-learn is a library designed for data modeling and implementing machine learning algorithms. One of the best things about Scikit-learn is how straightforward it is to use. You can get a machine learning model up and running with just a few lines of code. Plus, it's great for both beginners and experts — whether you're just starting with machine learning or fine-tuning a complex model, Scikit-learn makes the process easy."
          },
        ],
      },
    ]
  },
  { id: 6, 
    name: "PyTorch",
    sections: [
      { id: 0, 
        name: "6. PyTorch", 
        value: "PyTorch is an open-source Deep Learning framework based on the Torch library, designed for use with the Python programming language.",
        image: "",
        content: [
          { id: 1, 
            title: "",
            value: "PyTorch is an open-source deep learning framework that provides tools for building and training neural networks. It's known for its dynamic computation graph, which makes it flexible and intuitive to use."
          },
          { id: 2, 
            title: "",
            value: "PyTorch integrates seamlessly with Python, supports GPU acceleration, and includes an automatic differentiation system for efficient model training. It's widely used in both research and industry for developing advanced machine learning models."
          },
          { id: 3, 
            title: "",
            value: "In summary, PyTorch is a powerful, flexible, and user-friendly deep learning framework that has become one of the most popular tools for developing and deploying deep learning models."
          },
        ]
      },
      { id: 1,
        name: "6.1 PyTorch Features",
        value: "",
        image: pt_features,
        content: [
          { id: 1, 
            title: "Dynamic Computational Graphs",
            value: "PyTorch uses dynamic computational graphs. This means that the graph is built on the fly as operations are performed, allowing for more flexibility in model design and debugging. You can modify the graph during runtime, making it easier to work with complex models."
          },
          { 
            id: 2, 
            title: "Tensor Operations",
            value: "PyTorch supports multi-dimensional arrays (called tensors), similar to NumPy, but with the added capability of running on GPUs. This makes it suitable for large-scale computations needed for deep learning."
          },
          { id: 3, 
            title: "GPU Acceleration",
            value: "PyTorch is designed to take full advantage of GPUs for accelerated computation. By leveraging CUDA (NVIDIA's parallel computing platform), PyTorch allows for fast and efficient training of deep learning models."
          },
          { id: 4, 
            title: "Pre-built Models and Libraries",
            value: "PyTorch has a vast ecosystem, including libraries like TorchVision (for image processing), TorchText (for NLP), and TorchAudio (for audio processing). It also offers pre-built models through tools like torch.hub and torchvision.models."
          },
          { id: 5, 
            title: "Integration with Python",
            value: "PyTorch is fully integrated with Python, making it easy to use with other Python libraries like NumPy, SciPy, and Pandas. This makes it a preferred choice for many developers and researchers who are already familiar with Python."
          },
        ]
      },
      { id: 2,
        name: "6.2 Framework Trends",
        value: "",
        image: pt_trends,
        content: [
          { id: 1, 
            title: "",
            value: "This graph shows the trends in the usage of different machine learning frameworks based on the share of paper implementations from June 2020 to June 2024. These frameworks are marked with different color, and PyTorch is the most attractive one."
          },
        ]
      },
      { id: 3,
        name: "6.3 Companies with PyTorch",
        value: "PyTorch has become one of the most popular deep learning frameworks, and many leading companies across various industries are using it for their AI and machine learning applications. Here are some well-known companies that use PyTorch:",
        image: pt_companies,
        content: [
          { id: 1, 
            title: "Tesla",
            value: "Tesla utilizes PyTorch for a range of AI and deep learning tasks, particularly in the development of its Autopilot system and other autonomous driving technologies, such as the Full Self-Driving (FSD) system."
          },
          { 
            id: 2, 
            title: "NVIDIA",
            value: "NVIDIA supports PyTorch as part of its GPU-accelerated computing toolkit. Many deep learning models trained on NVIDIA GPUs use PyTorch, and NVIDIA contributes to optimizing PyTorch for its hardware."
          },
          { id: 3, 
            title: "Microsoft",
            value: "Microsoft uses PyTorch extensively in its Azure Machine Learning services and also contributes to its development. PyTorch is supported on Azure, making it easier for developers to deploy models in the cloud."
          },
          { 
            id: 4, 
            title: "Meta",
            value: "Facebook (now Meta) developed PyTorch, and it's heavily used across the company for various AI research and production tasks, including computer vision, natural language processing, and recommendation systems."
          },
          { id: 5, 
            title: "OpenAI",
            value: "OpenAI has used PyTorch for developing several of its models, including the GPT series (GPT-3 and beyond). PyTorch's flexibility and ease of use have made it a go-to framework for OpenAI's research."
          },
        ]
      }
    ]
  }, 
  { id: 7, 
    name: "Prerequisites",
    sections: [
      { id: 0, 
        name: "Python Programming Language",
        value: "",
        image: "",
        content: [
          { id: 1,
            title: "",
            value: "To learn PyTorch, it's important to have a basic understanding of Python. This includes concepts like variables, functions, classes, data types, and data structures. Python is relatively simple and intuitive, so if you're already familiar with other programming languages like Java or JavaScript, you should be able to pick it up quickly."
          },
        ],
      },
      { id: 1, 
        name: "Mathematical Fondations",
        value: "",
        image: "",
        content: [
          { id: 1, 
            title: "Linear Algebra",
            value: "A solid understanding of vectors, matrices, and operations like matrix multiplication is essential, as these concepts are fundamental to neural networks and many other machine learning algorithms. We'll explore these further in the next chapter, which focuses on Tensors."
          },
          { 
            id: 2, 
            title: "Calculus",
            value: "Familiarity with derivatives and integrals is crucial, especially when it comes to understanding how backPropagation and gradient descent optimize neural networks during training."
          },
          { id: 3, 
            title: "Probability and Statistics",
            value: "A good grasp of probability, random variables, and probability distributions (including the normal distribution) is important for understanding loss functions, activation functions, and the overall behavior of models."
          },
        ],
      },
    ]
  }, 
  { id: 8,
    name: "How to take this course?",
    sections: [
      { id: 0, 
        name: "8. How to take this course?",
        value: "",
        image: how_take_course,
        content: [
          { id: 1, 
            title: "Watch all Lessons from A to Z",
            value: "First and foremost, I recommend watching the entire course from beginning to the end, even if you're already familiar with PyTorch. I've designed this course to be concise and focused, so I won't waste our time on repetitive or irrelevant content. Make sure to watch every lesson."
          },
          { 
            id: 2, 
            title: "Take Notes",
            value: "While watching the lessons, I encourage you to take notes. Even if you prefer not to write extensively, jotting down key concepts and ideas can be very helpful. I strongly believe that writing things down aids in retaining new information. After each lesson, review your notes and practice the steps I demonstrated. This is the same approach I use to learn new material."
          },
          { id: 3, 
            title: "Replicate the Models",
            value: "Replicating models helps solidify the entire workflow, from building to training, through hands-on practice. Once mastered, this skill not only enhances problem-solving but also builds confidence in replicating and implementing new models from research papers."
          },
          { id: 4, 
            title: "Share your Work",
            value: "The more you share, the more you learn. To gain a deeper understanding of this course, try building your own project based on what you've learned and share it with others. Sharing your work allows others to provide feedback, which can be invaluable for improving your skills and creating better projects in the future. Be open to constructive criticism, as it will help you grow. Remember, I believe you can do even better than I have!"
          },
        ],
      },
    ]
  }
]