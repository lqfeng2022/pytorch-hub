import { Box, HStack, Image, Stack, Text } from '@chakra-ui/react'
import dlcat from '../../assets/dlcat.jpeg'
import mlcat from '../../assets/mlcat.jpeg'
import mldl from '../../assets/mldl.jpeg'


const DLComparing = () => {
  const comparing = {
    name: "3.2 Deep Learing V.S. Machine Learing",
    mlTitle: "1) Machine Learning Model (Decision Tree)", 
    dlTitle: "2) Deep Learning Model (Neural Network)",
    moreTitle: "3) Differences between Machine Learnig and Deep Learning",
    mlValues: [
      { id: 0, 
        content: "Here the input data must be structured, so we need to extract features from the image manually or use another algorithm to do, like color histograms, edge detection."
      },
      { id: 1, 
        content: "Then go through a decision tree model, the decision tree learns by splitting the data based on these input features.  It creates branches based on conditions (e.g., “if the image has a certain texture, go left, otherwise, go right”). The tree continues to split the data until it reaches a decision point, where it classifies the image as “cat” or “not a cat”."
      },
      { id: 2, 
        content: "For a new image, we gotta extract the same features and pass them through the decision tree, boring."
      },
    ],
    dlValues: [
      { id: 0, 
        content: "Here the input data is a raw image, an unstructured data, and no need for converting it to a structured data. Then go through a neural network, the neural network automatically extracts features, such as edges, textures, and patterns from the image, which in the initial layers. These features are then combined in deeper layers to detect more complex structures like fur patterns, eyes, and ears."
      },
      { id: 1, 
        content: "The neural network learns through backpropagation, adjusting its parameters based on the errors it makes during training on a labeled dataset (images labeled as “cat” or “not a cat”). The model gets better over time at identifying the features that distinguish a cat from other objects."
      },
      { id: 2, 
        content: "Once trained, the neural network can take a new image, process it through the layers, and output a probability score. Indicating whether the image is of a cat or not."
      },
    ],
    moreValues: [
      { id: 1, 
        name: "Structure Data",
        content: "Structured data is organized in a clear, predictable format, usually in tables with rows and columns, like a spreadsheet or a database. Each piece of data fits into a specific category, making it easy to search, analyze, and manage. Like numbers, dates, strings in Spreadsheets and Databases."
      },
      { id: 2, 
        name: "Unstructured Data",
        content: "Unstructured data is information that doesn't have a pre-defined or organized format. Unlike structured data, it doesn't fit neatly into tables, rows, or columns, making it harder to store, search, and analyze with traditional tools, like images, videos, audio recordings, text documents, webpages and so on."
      },
      { id: 3, 
        name: "Dataset",
        content: "Dataset is a collection of data, often organized in a structured format, that is used for analysis, research, or training machine learning models. Deep learning needs vast amounts  of data because they learn complex patterns, require lots of examples to generalize well, and often work with high-dimensional data. The more data the model has, the better it can perform, making large datasets essential for successful deep learning applications."
      },
      { id: 4, 
        name: "Algorithm",
        content: "Neural networks have become the most powerful models in the field of machine learning."
      },
      { id: 5, 
        name: "Supervised Learning",
        content: "Supervised learning is a type of machine learning where a model is trained on a labeled dataset. In this context, “labeled” means that each training example is paired with an output label, which represents the correct answer. The goal of supervised learning is to teach the model to make predictions or decisions based on new, unseen data. Supervised learning is all about teaching a model to predict or classify data by learning from examples where the correct answer is already known. It's one of the most common and powerful methods in machine learning, particularly useful when you have a large amount of labeled data to train on."
      },
      { id: 6, 
        name: "Unsupervised Learning",
        content: "Unsupervised learning is a type of machine learning where the model is trained on a dataset without explicit labels or output categories. Unlike supervised learning, the model is not given the correct answers or guidance during training. Instead, it tries to find patterns, relationships, or structures within the data on its own. Unsupervised learning is like exploring unknown territory— the model looks for patterns and relationships within data without being told what to find. It's particularly useful when you have a lot of data but don't know exactly what you're looking for, or when you want to uncover hidden structures or patterns within the data."
      },
    ],
  }

  return (
    <Box py={5}>
      <Text as='b' fontSize='lg'>{comparing.name}</Text>
      <Box py={5}>
        <Text as='b'>{comparing.mlTitle}</Text>
        <Image py={3} src={mlcat}/>
        <Stack spacing={4}>
          {comparing.mlValues.map((p) => <Text key={p.id}>{p.content}</Text>)}
        </Stack>
      </Box>
      <Box py={5}>
        <Text as='b'>{comparing.dlTitle}</Text>
        <Image py={3} src={dlcat}/>
        <Stack spacing={4}>
          {comparing.dlValues.map((p) => <Text key={p.id}>{p.content}</Text>)}
        </Stack>
      </Box>
      <Box py={5}>
        <Text as='b'>{comparing.moreTitle}</Text>
        <Image py={3} src={mldl}/>
        <Stack spacing={4}>
          {comparing.moreValues.map((p) =>
            <div key={p.id}>
              <HStack>
                <Text>{p.id}</Text>
                <Text as='b'>{p.name}</Text>
              </HStack>
              <Text>{p.content}</Text>
            </div>
          )}
        </Stack>
      </Box>
    </Box>
  )
}

export default DLComparing