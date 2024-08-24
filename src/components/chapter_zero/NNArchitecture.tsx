import { Box, Stack, Text, Image } from '@chakra-ui/react'
import nnmodel from '../../assets/nnmodel.jpeg'

const NNArchitecture = () => {
  const nnArchitecture = {
    name: "4.1 Neural Network Architecture",
    image: nnmodel,
    summary: "A neural network consists of layers of interconnected units called neurons or nodes. Here's a breakdown of its basic components:",
    values: [
      { id: 0, 
        title: "1) Neurons",
        content: "These are the fundamental processing units of the network, similar to neurons in the brain. Each neuron receives input, processes it (typically by applying a weighted sum followed by an activation function), and passes the output to the next layer. The activation function introduces non-linearity, allowing the network to model complex patterns."
      },
      { id: 1, 
        title: "2) Connections", 
        content: "Neurons in one layer are connected to neurons in the next layer through weights. These connections can be fully connected (where every neuron in one layer connects to every neuron in the next) or sparsely connected. The weights on these connections are the parameters that the network learns during training, adjusting to minimize the error in predictions."
      },
      { id: 2, 
        title: "3) Input", 
        content: "The first layer of the network, where data is fed into the network. Each neuron in this layer represents a feature of the input data. The number of neurons in this layer typically equals the number of input features."
      },
      { id: 3, 
        title: "4) Hidden Layers",
        content: "These are the layers between the input and output layers. Hidden layers perform computations and extract features from the input data. The term “deep learning” refers to networks with many hidden layers. The number of neurons and hidden layers can vary depending on the complexity of the task and are often fine-tuned as hyperparameters."
      },
      { id: 4, 
        title: "4) Output",
        content: "The final layer of the network that provides the output, such as a classification, prediction, or other result based on the processed input. The number of neurons in the output layer corresponds to the number of output classes in classification tasks or to the number of predicted values in regression tasks."
      },
    ]
  }

  return (
    <Box py={5}>
      <Text as='b' fontSize='lg'>{nnArchitecture.name}</Text>
      <Image py={5} src={nnArchitecture.image}/>
      <Stack spacing={4}>
        <Text fontSize='lg'>{nnArchitecture.summary}</Text>
        {nnArchitecture.values.map((p) => 
          <div key={p.id}>
            <Text as='b'>{p.title}</Text>
            <Text>{p.content}</Text>
          </div>
        )}
      </Stack>
    </Box>
  )
}

export default NNArchitecture