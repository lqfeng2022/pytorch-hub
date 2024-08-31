import { Center, Box, Stack, Text } from '@chakra-ui/react'
import Section from '../entities/Section'

interface Props {
  title: string,
  definition: Section
}
const Definition = ({ title, definition }: Props) => {
  return (
    <Box py={5}>
      <Text as='b' fontSize='xl' color='gray.600'>{title}</Text>
      <Center my={5} minH='250px' bg='red.50'>
        <Box maxW='500px'>
          <Text textAlign='center' fontSize='2xl' color='tomato'>{definition.value}</Text>
        </Box>
      </Center>
      <Stack spacing={4}>
        {definition.content.map((p) => <Text key={p.id}>{p.value}</Text>)}
      </Stack>
    </Box>
  )
}

export default Definition