import { Container, Divider, Flex, Heading, Stack, Text } from '@chakra-ui/react'
import introduce from '../data/introduce'
import FootLinks from '../components/FootLinks'
import sectionList from '../data/sectionList'

const AboutShape = () => {
  const intro = introduce[7]
  const { name: l, link: ll } = sectionList[13]
  // const { name: r, link: rl } = sectionList[0]

  return (
    <Container maxW='1200px' px='10px'>
      <Heading size='xl' color='gray.600' textAlign='center' py={3}>
        {intro.name}
      </Heading>
      <Stack spacing={4} pt={5}>
        {intro.content.map((p) => 
          <Text key={p.id} >{p.value}</Text>
        )}
      </Stack>
      <Flex align='center' h='80px'>
        <Divider variant='thick'/>
      </Flex>
      <FootLinks l={l} ll={ll}/>
    </Container>
  )
}

export default AboutShape