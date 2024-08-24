import { Box, SimpleGrid, Stack, Flex, Text, Image } from '@chakra-ui/react'
import ml from '../../data/ml'

const MLModels = () => {
  const content = {
    title: "2.2 Models in Machine Learning",
    value: "A machine learning model is basically a mathematical model that can make predictions or classifications on new data after it's been TRAINED on a dataset. There are many different types of models used in machine learning. Here, I'm gonna briefly introduce three of them for reference."
  }
  const svms = ml[2]
  const decisionTree = ml[3]
  const anns = ml[4]

  return (
    <Box py={5}>
      <Text as='b' fontSize='lg'>{content.title}</Text>
      <Text py={3} fontSize='lg'>{content.value}</Text>
      <Box pt={3}>
        <Text as='b'>{svms.name}</Text>
        <SimpleGrid columns={[1, null, 1]} spacing='10px' py={3}>
          <Flex alignItems='center' justifyContent='center'>
            <Image src={svms.image!}/>
          </Flex>
          <Stack my={2} spacing={2}>
            {svms.content.map((p) => <Text key={p.value}>{p.value}</Text>)}
          </Stack>
        </SimpleGrid>
      </Box>
      <Box>
        <Text as='b'>{decisionTree.name}</Text>
        <SimpleGrid columns={[1, null, 2]} spacing='10px' py={1}>
          <Stack my={2} spacing={2}>
            {decisionTree.content.map((p) => <Text key={p.value}>{p.value}</Text>)}
          </Stack>
          <Image src={decisionTree.image!}/>
        </SimpleGrid>
      </Box>
      <Box pt={5}>
        <Text as='b'>{anns.name}</Text>
        <SimpleGrid columns={[1, null, 2]} spacing='10px' py={3}>
          <Flex alignItems='center' justifyContent='center'>
            <Image src={anns.image!}/>
          </Flex>
          <Stack my={2} spacing={2}>
            {anns.content.map((p) => <Text key={p.value}>{p.value}</Text>)}
          </Stack>
        </SimpleGrid>
      </Box>
    </Box>
  )
}

export default MLModels