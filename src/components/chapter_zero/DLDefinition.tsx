import { Center, Box, Stack, Text } from '@chakra-ui/react'
import defintions from '../../data/definitions'

const DLDefinition = () => {
  const DLDefinition = defintions[2]

  return (
    <>
      <Center my={5} minH='250px' bg='red.50'>
        <Box maxW='500px'>
          <Text textAlign='center' fontSize='2xl' color='tomato'>{DLDefinition.value}</Text>
        </Box>
      </Center>
      <Stack spacing={4}>
        {DLDefinition.contents.map((p) => <Text key={p.id}>{p.name}</Text>)}
      </Stack>
    </>
  )
}

export default DLDefinition