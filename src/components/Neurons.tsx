import { Box, Stack, Text, Image } from '@chakra-ui/react'
import neurons from '../assets/neurons.jpeg'

const Neurons = () => {
  const nnArchitecture = {
    name: "4.2 What is Neurons in the Brain?",
    image: neurons,
    values: [
      { id: 0, 
        title: "Neurons",
        content: "Neurons are the fundamental working units of the brain and the nervous system. These specialized cells transmit information to each other through electrical and chemical signals. A single neuron can process signals from many other neurons and then pass the processed signal along to others. Each neuron consists of three main parts: the cell body (soma), dendrites (which receive signals), and an axon (which sends signals)."
      },
      { id: 1, 
        title: "Neural Network", 
        content: "Neurons are interconnected, forming complex neural networks that power all of the brain's functions, including seeing, hearing, smelling, tasting, feeling, thinking, and even dreaming. These networks are not static; they continuously change and adapt through a process called synaptic plasticity, which is essential for learning and memory."
      },
      { id: 2, 
        title: "THE NERVOUS SYSTEM", 
        content: "The nervous system, with the brain at its core, acts as the body's communication network. It controls everything we do, from breathing and moving to thinking, dreaming, and feeling. The nervous system includes not just the brain but also the spinal cord and peripheral nerves, which allow the brain to communicate with and control the rest of the body. This network manages both voluntary actions (like moving) and involuntary actions (like breathing)."
      },
      { id: 3, 
        title: "NNs (Brain) vs. NNs(Deep Learning)", 
        content: "The structure and function of neural networks in the brain have inspired the development of artificial neural networks in deep learning. While biological neural networks in the brain consist of billions of interconnected neurons that adapt and learn from experiences, deep learning neural networks are composed of artificial neurons arranged in layers. These artificial networks are designed to mimic the way the human brain processes information. In deep learning, neural networks automatically learn features and patterns from large datasets, similar to how the brain learns from sensory input. However, deep learning models, while powerful, are still much simpler and less flexible than the brain's incredibly complex networks."
      },
    ]
  }

  return (
    <Box py={5}>
      <Text as='b' fontSize='lg'>{nnArchitecture.name}</Text>
      <Image py={5} src={nnArchitecture.image}/>
      <Stack spacing={4}>
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

export default Neurons