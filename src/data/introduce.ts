export default [
  { id: 0, 
    name: "Introduction",
    value: "",
    image: "",
    content: [
      { id: 0,
        title: "",
        value: "The main goal of this book is to help beginners clearly understand Transformer architecture - the magic behind ChatGPT. I've built five deep learning models, starting simple (like a straight-line model) and gradually moving to more complex ones, following a general PyTorch workflow. By the end, we'll be building a language translation model using the Transformer. As we code together step by step, I'll break down the key concepts in math and computer science that power each model."
      },
    ]
  },
  { id: 1,
    name: "The Story Behind this Book",
    value: "",
    image: "",
    content: [
      { id: 0, 
        title: "",
        value: "What inspired me to write this book? Honestly, it all started with ChatGPT."
      },
      { id: 1,
        title: "",
        value: "Before ChatGPT-3.5 was released, I had a pretty standard view of artificial intelligence. Even though I had built a few basic machine learning models, I still believed AI was far from being truly ‘intelligent’. To me it seemed more like a tool for performing repetitive tasks. Even when AlphaGo defeated the world champion Go player in 2015, I didn’t see it as a breakthrough in intelligence. Since Go has clear well-defined rules, I assumed the machine was simply memorizing and processing these rules way faster than a human, which didn’t seem all that impressive. Look back, Maybe my perspective was limited because I didn’t fully understand what AI could really be."
      },
      { id: 2, 
        title: "",
        value: "When ChatGPT-3.5 came out on November 30, 2022, it was the first time AI truely impressed me with its human-like responses. When I asked ChatGPT questions, there were moments - just for a second - when I felt like who I wasn’t talking to a machine, but to a real person who understood what I was saying and gave suitable answers. Sometimes, though, it would provide responses that sounded convincing but were still wrong, delivered with a tone of certainty. It. Was almost like a person giving a wrong answer but insisting they were right."
      },
      { id: 3, 
        title: "",
        value: "In that sense, I realized it’s smarter than most of us in many ways. It made me wonder if, for many of us, the Turing Test might already be passed. That thought is both fascinating and a little unsetting. But with that, I believe we’ve on the verge of new opportunities and possibilities in the near future."
      },
      { id: 4, 
        title: "",
        value: "As they say, ‘Winter is coming’, and so is AI - and I’m coming along for the ride. I felt the need to understand why ChatGPT behaves so intelligently and what core algorithm powers it. This curiosity led me to dive deeper into the world of artificial intelligence. I came across the groundbreaking paper ‘Attention Is All You Need’, published by Google in 2017, which introduced the ‘Transformer’ architecture - the very foundation of ChatGPT."
      },
      { id: 5, 
        title: "",
        value: "Understanding this architecture is key to grasping how ChatGPT and similar large natural language models work. However, mastering the transformer architecture isn’t easy. To get there, I went through tons of Wikipedia entries, read well-known papers, and watched countless tutorials on the basic concepts and math behind deep learning."
      },
      { id: 6, 
        title: "",
        value: "But just doing some research, reading papers, or watching YouTube tutorials isn’t enough to truly master the Transformer. There’s no better approach to learn than through hands-on practice. So I started by building a series of deep learning models, starting from the most basic ones, like a simple linear regression. From there, I move on to slightly more complex models, like binary classification, a model for handwritten digit recognition, and eventually an image classification model using a Transformer encoder. Finally, we can tackle a language translation model, combining both the Transformer encoder and decoder."
      },
    ]
  },
  { id: 2,
    name: "The Content of this Book",
    value: "",
    image: "",
    content: [
      { id: 0, 
        title: "",
        value: "You can think of this book as a training course, the main goal is to guide you through building a language translation model based on the Transformer architecure. Once you’ve mastered this skill, you’ll be able to create many other models, like a words recognition model, other language translation models."
      },
      { id: 1, 
        title: "",
        value: "You can also push the boundaries and explore some new ideas. Imagine creating a model that translates a cat’s meow into human language, allowing us to understand what a cat is really saying. It would be a big project, and you might need a team for something like that. But the point is, as more people understand deep learning and know how to build a model, train a model, fine-tune a model, I beleve we’ll find partners to work together and create even more amazing models, doing somethings that seem almost like magic."
      },
      { id: 2, 
        title: "",
        value: "What a wonderful future that would be! We’ll be able to do things that once seemed impossible. One day, maybe we’ll naturally communicate with cats, dogs, birds or even plants - feeling what they feel and understanding nature more deeply than ever before."
      },
      { id: 3, 
        title: "",
        value: "In this book, I introduce artificial intelligence from the ground up, covering four main parts: the basic concepts of AI, tensors as a power tool for data manipulation, five deep learning models, and the mathematics behind these models."
      },
    ]
  },
  { id: 3,
    name: "Part 1: Introducing AI",
    value: "",
    image: "",
    content: [
      { id: 0, 
        title: "",
        value: "In this part, we’ll explore the history of artificial intelligence and how it evolved into ChatGPT, starting from early ideas like the Turning Test. As the field progressed, concepts like machine learing, deep learning, and neural network emerged. Since ChatGPT is a large language model based on deep learning, we’ll focus on understanding deep learning architecture and the essential knowledges surrounding it."
      },
      { id: 1, 
        title: "",
        value: "Next, we’ll introduce some popular and useful machine learning libraries and framework. For this book, we’ll use the PyTorch framework to build and train our deep learning model."
      },
    ]
  },
  { id: 4, 
    name: "Part 2: Tensor - A Data Manipulation Tool",
    value: "",
    image: "",
    content: [
      { id: 0,
        title: "",
        value: "Before building a deep learning model, we need to understand how to manipulate data, because data is like ingredients for a cook - essential for creating anything. In computere science , data represents everything in our world, from words, voices to images and videos. A tensor is a powerful tool for representing any kind of data with the complexity.",
      },
      { id: 1,
        title: "",
        value: "In this part, we’ll start by introducing tensors. A tensor is essentially a multi-dimentional array, and we’ll discuss important attributes, like its shape, number of dimensions, data type and the device it runs on. Next, we’ll dive into basic tensor operations, such as addition, subtraction, division, multiplication, and the slightly more complex matrix multiplication (which we’ll explain in detail). We’ll also explore aggregation operations, like finding the maximum or minimum value in a tensor.",
      },
      { id: 2, 
        title: "",
        value: "After that, we’ll focus one how to manipulate data using some useful PyTorch methods. For example, we can reshape a tensor, flatten it into a lower dimension, or expand it into a higher one. And we can stack two or more tensors together along a new dimension or concatenate them along an existing dimension. You’ll also learn how to change the data type and device of a tensor. These manipulation methods are incredibly useful throughout the entire process of building and training models."
      },
      { id: 3, 
        title: "",
        value: "We’ll also cover tensor indexing, which allows us to access any element, column, columns, row, rows or specific subset of data using filters. Additionally, we’ll touch on randomness, or more sepecifically, pseudorandomness in computer science. For reproducibility we can use a parameter called a random seed, which is an integer. Finally, we’ll discuss the device on which tensors run. By default, tensor urn on cpu, but if available, we can also use GPUs to speed up operations."
      },
    ]
  },
  { id: 5, 
    name: "Part 3: A Series of Deep Learning Models",
    value: "",
    image: "",
    content: [
      { id: 0, 
        title: "",
        value: "In this part, we’ll build a series deep learning models to explore the world of machine learning. This is he most exciting part of the book, because practice is the best teacher. By building really deep learning models, you’ll not only master essential skills but also gain a deeper understanding of complex concepts and mathematical functions. Simply knowing about functions and deep learning architectures isn’t enough to become proficient -  you need hands-on experience.",
      },
      { id: 1, 
        title: "",
        value: "Well, if I asked you to build a language tanslation model from the start, most of you would likely drop out. Complex tasks require a gradual approach, so we’ll begin with something simple. First, we’ll kick things off with a basic linear regression model, where the goal is to find a line that closely fit our target data. This will lay the groundwork.",
      },
      { id: 2, 
        title: "",
        value: "Next, we’ll move on to a slightly more complex model - a binary classification model. This model will help us distingush between two different groups of data points with a nested circular shape. In this project, we’ll introduce a non-linear function, which is crucial for understanding deep learning models.",
      },
      { id: 3, 
        title: "",
        value: "The 3rd one is a hnadwritten digit recoginiton model. We’ll build a convolutional neural network (CNN) to recogonize digits from 0 to 9. The key takeaway here is the CNN architecture, which is fundemental in computer vision tasks.",
      },
      { id: 4, 
        title: "",
        value: "Then we’ll create a Vision Transformer model, use to classfy images into 10 different categories. This will introduce you to the Transformer encoder, a groundbreaking architecture in deep learning.",
      },
      { id: 5, 
        title: "",
        value: " Finally, we’ll build a  language translation model using both transformer encoder and decoder - the same core architecture behind ChatGPT.",
      },
    ]
  },
  { id: 6,
    name: "Part 4: The Math Behind the Models",
    value: "",
    image: "",
    content: [
      { id: 0,
        title: "",
        value: "In this book, I introduce artificial intelligence from scratch with four main part, they are the basic concepts of artificial intelligence, tensor: a power tool to manimpulate data, then is about five deep learning models, and the the maths behind these models.",
      },
      { id: 1,
        title: "",
        value: "As we build thses models, we’ll use some important mathematical functions. After each model-building chaapter, we’ll explore the math behind the model, because understaning the math is key to truly grasping how the model works. We’ll also cover some basic computer sceince concepts along the way. I believe this is the best approach to learn the necessary mathematical concepts - when we encounter them in a project, we explore and understand them.",
      },
      { id: 2,
        title: "",
        value: "Of course, you could choose to study the math beforehand, but I prefere learning it as we go. There is so much math out there, so it’s more efficient to learn what you need  when you need it.",
      },
    ]
  },
  { id: 7,
    name: "About the Shape",
    value: "",
    image: "",
    content: [
      { id: 0,
        title: "",
        value: "After building all five deep learning models, what I want to empasize one thing: the shape. As we've seen,  the shape of tensors is a key attribute in machine learning, and it plays a critical role in how models process data. So throughout this book, whenever I build a model, I always draw the shape of the data at every stage, from the input, through the hidden layers, all the way to the output. This helps clarify how data flows and transforms within the model."
      },
      { id: 1, 
        title: "",
        value: "The reason I do this is simple: I want to visualize the model's architecture so we can clearly see what the model is doing at each step. And this becomes pretty important as our model grow more complex. Computers love data, while people prefer pictures. By assigning a shape to the data, we can gain a more intuitive understanding for the entire architecture. It makes it easier to grasp the ideas behind the model, compared tojust looking at raw numbers and technical descriptions."
      },
      { id: 2, 
        title: "",
        value: "In this course, after building each model, we also dive into the mathematics behind them, and I believe this is the most challenging part of the training. Building and training model is a bit like following a manufacturing process to create a product. But if you want to improve the quantity of that product, you will need to enhance your model, possibly by reworking its architecture. And to do that effectively, you must first understand the math that underpins your model. Math is the key, you need to find it, discover and master it, then you can apply it repeatedly to improve your model."
      },
      { id: 3, 
        title: "",
        value: "Well, now we already known that math is the key to the door of machine learnig, so how do we learn and master it? Do we need to remember all those fancy formulas? No, it's the worthest approach, and even you rember the formulas, you don't understand the meaning behind those formulas, you still cannot use them properly. Instead, you should focus on how to understand them, Here I think we don't need to rembember the fancy formulas, you just need to know it, like known the key parameters how to influence the formula. In today's AI-drive world, especially with tools like ChatGPT,  memorizing formulas isn't as important. What truely matters is understanding the meaning behind these functions and knowing how to apply them to specific tasks, like building a deep learning model, that's the real point."
      },
      { id: 4, 
        title: "",
        value: "So in our maths course, I visualize each formula with a shape. These shapes might be a bell-shaped, S-shaped, downhill and so on. It's easy to visualize these shapes, and when you associate the key parameters of the formulas with their shapes, you can quicky grasp the meaning behind them. For example, consider the normal distribution, often represented by a bell-shape, or as I prefer, a mountain-shape. One parameter, the mean, determines the location of the mountain, while the variance shapes it. Imagine a steep, narrow mountain with a small variance compare to something like Mount Fuji, which has a broadeer, smoother slope due to a large variance."
      },
    ]
  },
]