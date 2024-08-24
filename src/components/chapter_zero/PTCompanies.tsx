import { Box, Stack, Text, Image } from '@chakra-ui/react'
import ptcompanies from '../../assets/companies.jpeg'

const PTCompanies = () => {
  const companies = {
    name: "6.3 Companies with PyTorch",
    content: "PyTorch has become one of the most popular deep learning frameworks, and many leading companies across various industries are using it for their AI and machine learning applications. Here are some well-known companies that use PyTorch:",
    image: ptcompanies,
    values: [
      { id: 1, 
        title: "Tesla",
        value: "Tesla use PyTorch for various AI and deep learning tasks, particularly in the development of its Autopilot system and other autonomous driving technologies."
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

  return (
    <Box py={5}>
      <Text as='b' fontSize='lg'>{companies.name}</Text>
      <Text fontSize='lg' pt={3}>{companies.content}</Text>
      <Image py={5} src={companies.image}/>
      <Stack spacing={4}>
        {companies.values.map((p) => 
          <div key={p.id}>
            <Text as='b'>{p.title}</Text>
            <Text>{p.value}</Text>
          </div>
        )}
      </Stack>
    </Box>
  )
}

export default PTCompanies