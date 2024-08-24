import { Box, Center, Stack, Text, Image } from '@chakra-ui/react';
import MLModels from './MLModels'
import ml from '../../data/ml'

const MLSection = () => {
  const mlDefinition = ml[0]
  const comparing = ml[1]

  return (
    <>
      {/* 2. Machine Learning */}
      <Text as='b' fontSize='xl'>2. Machine Learning</Text>
      {/* ML definition */}
      <Box>
        <Center my={5} minH='250px' bg='red.50'>
          <Box maxW='500px'>
            <Text textAlign='center' fontSize='2xl' color='tomato'>{mlDefinition.value}</Text>
          </Box>
        </Center>
        <Stack spacing={4}>
          {mlDefinition.content.map((p) => <Text key={p.id}>{p.value}</Text>)}
        </Stack>
      </Box>
      {/* 2.1 Machine Learing V.S. Traditinal Programming */}
      <Box py={5}>
        <Text as='b' fontSize='lg'>{comparing.name}</Text>
        <Image py={5} src={comparing.image!}/>
        <Stack spacing={4}>
          {comparing.content.map((p) => 
            <div key={p.id}>
              <Text as='b'>{p.title}</Text>
              <Text>{p.value}</Text>
            </div>
          )}
        </Stack>
      </Box>
      <MLModels/>
    </>
  )
}

export default MLSection