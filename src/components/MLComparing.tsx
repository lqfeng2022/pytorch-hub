import { Box, Image, Stack, Text } from '@chakra-ui/react'
import mlcomparing from '../assets/mlcomparing.jpeg'

const MLComparing = () => {
  const comparing = {
    name: "2.1 Machine Learing V.S. Traditinal Programming",
    image: mlcomparing,
    values: [
      { id: 0, 
        title: "1) Traditional Programming", 
        content: "Traditional programming relies on rule-based coding, where you explicitly define the rules for the computer to follow. In this approach, you determine the logic and sequence of operations, and the computer simply executes those instructions."
      },
      { id: 1, 
        title: "2) Machine Learning",
        content: "Machine Learning, on the other hand, is a data-driven approach. Instead of writing explicit rules, you provide a model and a large dataset. The model is then trained to learn patterns and relationships between the input and output on its own."
      },
    ]
  }

  return (
    <Box py={5}>
      <Text as='b' fontSize='lg'>{comparing.name}</Text>
      <Image py={5} src={comparing.image}/>
      <Stack spacing={4}>
        {comparing.values.map((p) => 
          <div key={p.id}>
            <Text as='b'>{p.title}</Text>
            <Text>{p.content}</Text>
          </div>
        )}
      </Stack>
    </Box>
  )
}

export default MLComparing