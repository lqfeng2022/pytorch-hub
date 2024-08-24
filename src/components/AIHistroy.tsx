import { Box, Flex, Image, SimpleGrid, Stack, Text } from '@chakra-ui/react';
import turningTest from '../assets/TuringTest.png'
import expertSystem from '../assets/expertSystem.jpeg'
import connectionism from '../assets/Connectionism.jpeg'
import cnn from '../assets/cnn.jpeg'
import alphago from '../assets/alphago.jpeg'
import openai from '../assets/openai.jpeg'
import aihistory from '../assets/aihistory.jpeg'


const AIHistroy = () => {
  const contents = [
    { id: 1, 
      name: "1) Turning Test", 
      content: [
        { id: 0, 
          value: "Alan Turing was the first to really explore what he called 'machine intelligence'. In his 1950 paper, 'Computing Machinery and Intelligence', he introduced the Turing test and argued that machines could actually be intelligent. This was a major milestone and laid the foundation for AI research, which officially became an academic field in 1956."
        },
        { id: 1, 
          value: "The typical way to understand the Turing test is this: there are 3 players—A, B, and C. Player C is the interrogator, and his job is to figure out which of the other two, A or B, is a computer and which one is a human. The tricky part? The interrogator can only ask written questions to make that decision."
        }
    ]},
    { id: 2, 
      name: "2) Expert System", 
      content: [
        { id: 0, 
          value: "In the early 1980s, AI research got a boost thanks to the commercial success of expert systems - AI programs - that mimicked the knowledge and analytical skills of human experts."
        },
    ]},
    { id: 3, 
      name: "3) Connectionism + Neural Network", 
      content: [
        { id: 0, 
          value: "A few years later, one of the biggest breakthroughs was the revival of CONNECTIONISM, led by Geoffrey Hinton and others, which brought neural network research back into the spotlight — it's all about simulating how the brain's neural networks work."
        },
        { id: 1, 
          value: "The core idea behind connectionism is that mental processes can be understood as networks that interconnected units. These units represent neurons, and the connections between them act like synapses, just like in the human brain."
        }
    ]},
    { id: 4, 
      name: "4) Convolutional Neural Network", 
      content: [
        { id: 0, 
          value: "In 1990, Yann LeCun showed that convolutional neural networks could accurately recognize handwritten digits. This was one of the first major successes for neural networks."
        }
    ]},
    { id: 5, 
      name: "5) AlphaGo", 
      content: [
      { id: 0,
        value: "Deep learning started taking over industry benchmarks in 2012, thanks to better hardware and access to huge amounts of data."
      },
      { id: 1,
        value: "Then, in 2015, DeepMind's AlphaGo made headlines by beating the world champion Go player. The program was only taught the rules and figured out its own strategy."
      }
    ]},
    { id: 6, 
      name: "6) ChatGPT", 
      content: [
      { id: 0, 
        value: "Fast forward to the end of 2022, and OpenAI released GPT-3, a large language model that can generate text that feels like it was written by a human."
      }
    ]},
  ]

  return (
    <Box py={5}>
      <Text as='b' fontSize='lg'>1.2 The History of AI</Text>
      <Image py={5} src={aihistory}/>
      {/* 1)Turning Test */}
      <SimpleGrid columns={[1, null, 2]} spacing='20px' py={3}>
        <Box>
          <Text as='b'>{contents[0].name}</Text>
          <Stack my={2} spacing={2}>
            {contents[0].content.map((p) => <Text key={p.value}>{p.value}</Text>)}
          </Stack>
        </Box>
        <Flex alignItems='center' justifyContent='center'>
          <Image src={turningTest}/>
        </Flex>
      </SimpleGrid>
      {/* 2)Expert System */}
      <SimpleGrid columns={[2, null, 2]} spacing='20px' py={3}>
        <Box>
          <Text as='b'>{contents[1].name}</Text>
          <Stack my={2} spacing={2}>
            {contents[1].content.map((p) => <Text key={p.value}>{p.value}</Text>)}
          </Stack>
        </Box>
        <Box maxW='200px'>
          <Image src={expertSystem}/>
        </Box>
      </SimpleGrid>
      {/* 3) Connectionism + Neural Network */}
      <SimpleGrid columns={[1, null, 2]} spacing='20px' py={3}>
        <Flex alignItems='center' justifyContent='center'>
          <Image src={connectionism}/>
        </Flex>
        <Box>
          <Text as='b'>{contents[2].name}</Text>
          <Stack my={2} spacing={2}>
            {contents[2].content.map((p) => <Text key={p.value}>{p.value}</Text>)}
          </Stack>
        </Box>
      </SimpleGrid>
      {/* 4) Convolutional Neural Network */}
      <SimpleGrid columns={[1, null, 1]} spacing='20px' py={3}>
        <Flex alignItems='center' justifyContent='center'>
          <Image src={cnn}/>
        </Flex>
        <Box>
          <Text as='b'>{contents[3].name}</Text>
          <Stack my={2} spacing={2}>
            {contents[3].content.map((p) => <Text key={p.value}>{p.value}</Text>)}
          </Stack>
        </Box>
      </SimpleGrid>
      {/* 5)AlphaGo */}
      <SimpleGrid columns={[1, null, 2]} spacing='20px' py={3}>
        <Flex alignItems='center' justifyContent='center'>
          <Image src={alphago}/>
        </Flex>
        <Box>
          <Text as='b'>{contents[4].name}</Text>
          <Stack my={2} spacing={2}>
            {contents[4].content.map((p) => <Text key={p.value}>{p.value}</Text>)}
          </Stack>
        </Box>
      </SimpleGrid>
      {/* 6)ChatGPT */}
      <SimpleGrid columns={[1, null, 2]} spacing='20px' py={3}>
        <Box>
          <Text as='b'>{contents[5].name}</Text>
          <Stack my={2} spacing={2}>
            {contents[5].content.map((p) => <Text key={p.value}>{p.value}</Text>)}
          </Stack>
        </Box>
        <Flex alignItems='center' justifyContent='center'>
          <Image src={openai}/>
        </Flex>
      </SimpleGrid>
    </Box>
  )
}

export default AIHistroy