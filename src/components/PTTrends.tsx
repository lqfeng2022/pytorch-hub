import { Box, Stack, Text, Image } from '@chakra-ui/react'
import pttrends from '../assets/pytorchtrends.jpeg'

const PTTrends = () => {
  const trends = {
    name: "6.2 Trends",
    image: pttrends,
    values: [
      { id: 1, 
        value: "This graph shows the trends in the usage of different machine learning frameworks based on the share of paper implementations from March 2020 to June 2024. These frameworks are marked with different color, and PyTorch is the most attractive one."
      },
      { 
        id: 2, 
        value: "The red area (representing PyTorch) shows significant growth over time. Starting from March 2020, PyTorch's share of implementations steadily increased, eventually becoming the dominant framework by the middle of 2021. This trend continues with PyTorch maintaining the largest share through to June 2024."
      }
    ]
  }

  return (
    <Box py={5}>
      <Text as='b' fontSize='lg'>{trends.name}</Text>
      <Image py={5} src={trends.image}/>
      <Stack spacing={4}>
        {trends.values.map((p) => 
          <div key={p.id}>
            <Text>{p.value}</Text>
          </div>
        )}
      </Stack>
    </Box>
  )
}

export default PTTrends