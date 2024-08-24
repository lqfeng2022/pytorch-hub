import { Box, SimpleGrid, Stack, Flex, Text, Image } from '@chakra-ui/react'
import svms from '../assets/svms.jpeg'
import ann from '../assets/ann.jpeg'
import decisionTree from '../assets/decisionTree.jpeg'

const MLModels = () => {
  const mlModels = [
    { id: 0, 
      name: "1) Support-vector machines (SVMs)", 
      content: [
        { id: 0, 
          value: "SVMs, or Support Vector Machines, a type of supervised learning method that used for both classification and regression tasks. When you have a set of training examples, each labeled as belonging to one of two categories, an SVM algorithm builds a model to predict which category a new example will fall into."
        },
        { id: 1, 
          value: "Not only the linear regression, SVMs can also handle non-linear classification."
        }
    ]},
    { id: 1, 
      name: "2) Decision Tree",
      content: [
        { id: 0, 
          value: "It is a hierarchical model that uses a tree-like structure to represent decisions and their possible consequences. It's more like a human-logic simulation than the biological mimics of ANNs.A decision tree is like a chat flow, where each internal note represents a TEST on an attribute, each branch represents the output of the test, and each leaf node represents a class label. The paths from the root to the leaves show the classification rules."
        },
    ]},
    { id: 2, 
      name: "3) Artificial Neural Networks (ANNs)", 
      content: [
        { id: 0, 
          value: "It's a connectionist system inspired by the animal brain - especially the human brain. The brain is basically a network of neurons that transmit electrical signals."
        },
        { id: 1, 
          value: "An Artificial Neural Network (ANN) mimics this by using a collection of connected units or 'Artificial Neurons', similar to how brain neurons work. Each artificial neuron in this model receive signals, processes them, and then sends them on to the other neurons."
        }
    ]},
  ]

  return (
    <Box py={5}>
      <Text as='b' fontSize='lg'>2.2 Models in Machine Learning</Text>
      <Text py={3} fontSize='lg'>A machine learning model is basically a mathematical model that can make predictions or classifications on new data after it's been TRAINED on a dataset. There are many different types of models used in machine learning. Here, I'm gonna briefly introduce three of them for reference.</Text>
      <Box pt={3}>
        <Text as='b'>{mlModels[0].name}</Text>
        <SimpleGrid columns={[1, null, 1]} spacing='10px' py={3}>
          <Flex alignItems='center' justifyContent='center'>
            <Image src={svms}/>
          </Flex>
          <Stack my={2} spacing={2}>
            {mlModels[0].content.map((p) => <Text key={p.value}>{p.value}</Text>)}
          </Stack>
        </SimpleGrid>
      </Box>
      <Box>
        <Text as='b'>{mlModels[1].name}</Text>
        <SimpleGrid columns={[1, null, 2]} spacing='10px' py={1}>
          <Stack my={2} spacing={2}>
            {mlModels[1].content.map((p) => <Text key={p.value}>{p.value}</Text>)}
          </Stack>
          <Image src={decisionTree}/>
        </SimpleGrid>
      </Box>
      <Box pt={5}>
        <Text as='b'>{mlModels[2].name}</Text>
        <SimpleGrid columns={[1, null, 2]} spacing='10px' py={3}>
          <Flex alignItems='center' justifyContent='center'>
            <Image src={ann}/>
          </Flex>
          <Stack my={2} spacing={2}>
            {mlModels[2].content.map((p) => <Text key={p.value}>{p.value}</Text>)}
          </Stack>
        </SimpleGrid>
      </Box>
    </Box>
  )
}

export default MLModels