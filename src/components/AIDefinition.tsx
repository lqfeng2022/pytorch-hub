import { Center, Box, Text, Stack } from '@chakra-ui/react'
import defintions from '../data/definitions'

const AIDefinition = () => {
  const aiDefinition = defintions[0]

  return (
    <>
      <Center my={5} minH='250px' bg='red.50'>
        <Box maxW='500px'>
          <Text textAlign='center' fontSize='2xl' color='tomato'>{aiDefinition.value}</Text>
        </Box>
      </Center>
      <Stack spacing={4}>
        {aiDefinition.contents.map((p) => <Text key={p.id}>{p.name}</Text>)}
      </Stack>
    </>
  )
}

export default AIDefinition