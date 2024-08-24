import { Box, Center, Image, Stack, Text } from '@chakra-ui/react';
import AIHistroy from './AIHistroy';
import ai from '../../data/ai'

const AISection = () => {
  const whats =  ai[0]
  const reason = ai[1]

  return (
    <>
      <Text as='b' fontSize='xl'>1. Artificial Intelligence</Text>
      {/* AI Definition */}
      <Box py={5}>
        <Center my={5} minH='250px' bg='red.50'>
          <Box maxW='500px'>
            <Text textAlign='center' fontSize='2xl' color='tomato'>{whats.value}</Text>
          </Box>
        </Center>
        <Stack spacing={4}>
          {whats.content?.map((p) => <Text key={p.id}>{p.value}</Text>)}
        </Stack>
      </Box>
      {/* 1.1 Why we should know about AI? */}
      <Box py={5}>
        <Text as='b' fontSize='lg'>{reason.name}</Text>
        <Image py={5} src={reason.image!}/>
        <Stack spacing={4}>
          {reason.content?.map((p) => 
            <div key={p.id}>
              <Text as='b'>{p.title}</Text>
              <Text>{p.value}</Text>
            </div>
          )}
        </Stack>
      </Box>
      <AIHistroy/>
    </>
  )
}

export default AISection