import { Center, Box, Stack, Text } from '@chakra-ui/react'
import definitions from '../data/definitions'

const PTDefinition = () => {
  const ptdefinition = definitions[4]

  return (
    <>
      <Center my={5} minH='250px' bg='red.50'>
        <Box maxW='500px'>
          <Text textAlign='center' fontSize='2xl' color='tomato'>{ptdefinition.value}</Text>
        </Box>
      </Center>
      <Stack spacing={4}>
        {ptdefinition.contents.map((p) => <Text key={p.id}>{p.name}</Text>)}
      </Stack>
    </>
  )
}

export default PTDefinition