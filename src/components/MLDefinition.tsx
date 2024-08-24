import { Box, Center, Stack, Text } from '@chakra-ui/react'
import defintions from '../data/definitions'

const MLDefinition = () => {
  const mlDefinition = defintions[1]

  return (
    <>
      <Center my={5} minH='250px' bg='red.50'>
        <Box maxW='500px'>
          <Text textAlign='center' fontSize='2xl' color='tomato'>{mlDefinition.value}</Text>
        </Box>
      </Center>
      <Stack spacing={4}>
        {mlDefinition.contents.map((p) => <Text key={p.id}>{p.name}</Text>)}
      </Stack>
    </>
  )
}

export default MLDefinition