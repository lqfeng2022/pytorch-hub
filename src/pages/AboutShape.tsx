import { Container, Heading, Stack, Text } from '@chakra-ui/react'
import introduce from '../data/introduce'

const AboutShape = () => {
  const intro = introduce[7]

  return (
    <Container maxW='1200px' px='10px'>
      <Heading size='xl' color='gray.600' textAlign='center' py={3}>
        {intro.name}
      </Heading>
      <Stack spacing={4}>
        {intro.content.map((p) => 
          <Text key={p.id} >{p.value}</Text>
        )}
      </Stack>
    </Container>
  )
}

export default AboutShape