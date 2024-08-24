import { Box, SimpleGrid, Stack, Flex, Text, Image } from '@chakra-ui/react'
import relationships from '../../assets/relationships.jpeg'

const DLRelationship = () => {
  const comparing = {
    name: "3.1 Deep Learing < Machine Learing < Artificial Intelligence",
    image: relationships,
    summary: "In summary,  AI is the overall goal of creating intelligent systems. ML is a way to achieve AI by enabling machines to learn from data. DL is a more advanced technique within ML that uses layered neural networks to learn from large amounts of data.",
    values: [
      { id: 0, 
        title: "1) Artificial Intelligence (AI)", 
        content: "Artificial Intelligence (AI) is the broadest concept, referring to the development of computer systems that can perform tasks typically requiring human intelligence. This includes things like problem-solving, understanding language, recognizing patterns, and making decisions."
      },
      { id: 1, 
        title: "2) Machine Learning (ML)",
        content: "Machine Learning (ML) is a subset of AI. It focuses on creating algorithms and models that allow computers to learn from and make predictions or decisions based on data. Instead of being explicitly programmed to perform a task, a machine learning model is trained on data to find patterns and improve over time."
      },
      { id: 2, 
        title: "3) Deep Learning (DL)",
        content: "Deep Learning (DL) is a subset of Machine Learning. It involves neural networks with many layers (hence DEEP) that can automatically learn complex patterns in data. Deep Learning excels at tasks like image and speech recognition, where traditional machine learning methods might struggle."
      },
    ]
  }

  return (
    <Box py={5}>
      <Stack spacing={3}>
        <Text as='b' fontSize='lg'>{comparing.name}</Text>
        <Text as='i' fontSize='lg'>{comparing.summary}</Text>
      </Stack>
      {/* 1)Turning Test */}
      <SimpleGrid columns={[1, null, 2]} spacing='20px' py={3}>
        <Flex alignItems='center' justifyContent='center'>
          <Image src={comparing.image}/>
        </Flex>
        <Stack spacing={4}>
          {comparing.values.map((p) => 
            <div key={p.id}>
              <Text as='b'>{p.title}</Text>
              <Text>{p.content}</Text>
            </div>
          )}
        </Stack>
      </SimpleGrid>
    </Box>
  )
}

export default DLRelationship