import { Box, Stack, Text, Image } from '@chakra-ui/react'
import libraries from '../assets/libraries.jpeg'

const Libraries = () => {
  const librariesAI = {
    image: libraries,
    values: [
      { id: 1, 
        title: "PyTorch",
        content: [
          { id: 1,
            name: "PyTorch is a popular open-source deep learning framework based on the Torch library, developed by Meta AI. It's incredibly user-friendly, making it a go-to tool for building and training neural networks, especially if you're looking for something intuitive and flexible."
          },
          { id: 2, 
            name: "Widely used in both research and industry, PyTorch is powerful yet accessible. Whether you're a beginner in deep learning or working on advanced AI projects, PyTorch is a great choice. It's also backed by a strong community, so you'll find plenty of tutorials, resources, issue trackers, and support available."
          }
        ],
      },
      { id: 2, 
        title: "TensorFlow",
        content: [
          { id: 1,
            name: "TensorFlow is one of the most popular deep learning frameworks, developed by the Google Brain team. It's more structured than PyTorch, making it ideal for scaling up projects and deploying them in production, especially when you need to run them across multiple devices or in the cloud."
          },
          { id: 2, 
            name: "One of the standout features of TensorFlow is its extensive collection of built-in tools and libraries, allowing you to create anything from simple neural networks to complex, state-of-the-art models. With TensorFlow Lite and TensorFlow.js, you can even run your models on mobile devices or in the browser."
          },
          { id: 3, 
            name: "While TensorFlow can be challenging to learn initially, once you're up and running, it proves to be incredibly versatile. Whether you're just starting out or building advanced models, TensorFlow is a solid choice, backed by a massive community and abundant resources to support your journey."
          }
        ],
      },
      { id: 3, 
        title: "NumPy",
        content: [
          { id: 1,
            name: "NumPy is one of the most widely used scientific computing libraries, designed to perform a variety of mathematical operations on arrays and matrices. Whether you're working with matrices, statistics, or just large sets of numbers, NumPy is an essential tool."
          },
          { id: 2, 
            name: "It serves as the foundation for many other data science libraries, such as Pandas, TensorFlow, and PyTorch. Getting comfortable with NumPy is a crucial step if you're venturing into data science or machine learning. What makes NumPy stand out is its efficiency. It allows you to perform complex calculations on large datasets with remarkable speed. Additionally, it offers a wide range of built-in functions for tasks like linear algebra, random number generation, and much more."
          },
        ],
      },
      { id: 4, 
        title: "MatPLotLib",
        content: [
          { id: 1,
            name: "Matplotlib is a powerful data visualization library used for creating a wide range of plots and graphs. Whether you need a simple line graph, a detailed bar chart, or even a complex plot, Matplotlib has you covered. It's incredibly versatile, allowing you to create almost any type of chart or graph you can imagine."
          },
        ],
      },
      { id: 5, 
        title: "Scikit-Learn",
        content: [
          { id: 1,
            name: "Scikit-learn is a library designed for data modeling and implementing machine learning algorithms. One of the best things about Scikit-learn is how straightforward it is to use. You can get a machine learning model up and running with just a few lines of code. Plus, it's great for both beginners and experts — whether you're just starting with machine learning or fine-tuning a complex model, Scikit-learn makes the process easy."
          },
        ],
      },
    ]
  }

  return (
    <Box py={3}>
      <Image py={3} src={librariesAI.image}/>
      <Stack spacing={4}>
        {librariesAI.values.map((p) => 
          <div key={p.id}>
            <Text as='b'>{p.title}</Text>
            {p.content.map((t) => <Text py={2}>{t.name}</Text>) }
          </div>
        )}
      </Stack>
    </Box>
  )
}

export default Libraries