import { Box, Flex, Image, SimpleGrid, Stack, Text } from '@chakra-ui/react';
import aihistory from '../../assets/aihistory.jpeg';
import ai from '../../data/ai';

const AIHistroy = () => {
  const turningTest = ai[2]
  const expertSystem = ai[3]
  const connectionism = ai[4]
  const cnn = ai[5]
  const alphago = ai[6]
  const openai = ai[7]

  return (
    <Box py={5}>
      <Text as='b' fontSize='lg'>1.2 The History of AI</Text>
      <Image py={5} src={aihistory}/>
      {/* 1)Turning Test */}
      <Box>
        <Text as='b'>{turningTest.name}</Text>
        <SimpleGrid columns={[1, null, 2]} spacing='20px' py={3}>
          <Flex alignItems='center' justifyContent='center'>
            <Image src={turningTest.image!}/>
          </Flex>
          <Stack my={2} spacing={2}>
            {turningTest.content.map((p) => <Text key={p.value}>{p.value}</Text>)}
          </Stack>
        </SimpleGrid>
      </Box>
      {/* 2)Expert System */}
      <Box>
        <Text as='b'>{expertSystem.name}</Text>
        <SimpleGrid columns={[2, null, 2]} spacing='20px' py={3}>
          <Stack my={2} spacing={2}>
            {expertSystem.content.map((p) => <Text key={p.value}>{p.value}</Text>)}
          </Stack>
          <Box maxW='200px'>
            <Image src={expertSystem.image!}/>
          </Box>
        </SimpleGrid>
      </Box>
      {/* 3) Connectionism + Neural Network */}
      <Box>
        <Text as='b'>{connectionism.name}</Text>
        <SimpleGrid columns={[1, null, 2]} spacing='20px' py={3}>
          <Flex alignItems='center' justifyContent='center'>
            <Image src={connectionism.image!}/>
          </Flex>
          <Stack my={2} spacing={2}>
            {connectionism.content.map((p) => <Text key={p.value}>{p.value}</Text>)}
          </Stack>
        </SimpleGrid>
      </Box>
      {/* 4) Convolutional Neural Network */}
      <Box>
        <Text as='b'>{cnn.name}</Text>
        <SimpleGrid columns={[1, null, 1]} spacing='20px' py={3}>
          <Stack my={2} spacing={2}>
            {cnn.content.map((p) => <Text key={p.value}>{p.value}</Text>)}
          </Stack>
          <Flex alignItems='center' justifyContent='center'>
            <Image src={cnn.image!}/>
          </Flex>
        </SimpleGrid>
      </Box>
      {/* 5)AlphaGo */}
      <Box>
        <Text as='b'>{alphago.name}</Text>
        <SimpleGrid columns={[1, null, 2]} spacing='20px' py={3}>
          <Flex alignItems='center' justifyContent='center'>
            <Image src={alphago.image!}/>
          </Flex>
          <Stack my={2} spacing={2}>
            {alphago.content.map((p) => <Text key={p.value}>{p.value}</Text>)}
          </Stack>
        </SimpleGrid>
      </Box>
      {/* 6)ChatGPT */}
      <Box>
        <Text as='b'>{openai.name}</Text>
        <SimpleGrid columns={[1, null, 2]} spacing='20px' py={3}>
          <Stack my={2} spacing={2}>
            {openai.content.map((p) => <Text key={p.value}>{p.value}</Text>)}
          </Stack>
          <Flex alignItems='center' justifyContent='center'>
            <Image src={openai.image!}/>
          </Flex>
        </SimpleGrid>
      </Box>
    </Box>
  )
}

export default AIHistroy