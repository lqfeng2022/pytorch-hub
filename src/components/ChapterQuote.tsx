import { Center, Stack, Text } from '@chakra-ui/react'

const ChapterQuote = () => {
  const quote = {
    content: "A computer would deserve to be called intelligent if it could deceive a human into believing that it was human.", 
    author: "Alan Turing wrote in 1950"
  }

  return (
    <Center py={3}>
      <Stack maxW='250px'>
        <Text as='i' textAlign='center' pb={3}>"{quote.content}"</Text>
        <Text as='i' textAlign='center'>— {quote.author}</Text>
      </Stack>
    </Center>
  )
}

export default ChapterQuote