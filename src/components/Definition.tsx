import { Center, Box, Stack, Text } from '@chakra-ui/react'
import Section from '../entities/Section'

interface Props {
  definition: Section
}
const Definition = ({ definition }: Props) => {
  return (
    <>
      <Center my={5} minH='250px' bg='red.50'>
        <Box maxW='500px'>
          <Text textAlign='center' fontSize='2xl' color='tomato'>{definition.value}</Text>
        </Box>
      </Center>
      <Stack spacing={4}>
        {definition.content.map((p) => <Text key={p.id}>{p.value}</Text>)}
      </Stack>
    </>
  )
}

export default Definition