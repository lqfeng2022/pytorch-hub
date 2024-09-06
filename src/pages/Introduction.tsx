import { Container, Text, Stack, Heading, Divider, Flex } from '@chakra-ui/react'
import Section from '../entities/Section'
import introduce from '../data/introduce'
import IntroduceGrid from '../components/IntroduceGrid'
import FootLinks from '../components/FootLinks'

const Introduction = () => {
  const intro = introduce[0]
  const [ story, contents, ai, tensor, models, maths ] = introduce.slice(1, 7)

  const footer = {
    l: "Book Cover",
    r: "0. Artificial Intelligence",
    ll: "/",
    rl: "/artificial-intelligence"
  }

  return (
    <Container maxW='1200px' px='10px'>
      <Heading size='xl' color='gray.600' textAlign='center' py={3}>
        {intro.name}
      </Heading>
      <Text py={5}>{intro.content[0].value}</Text>
      <IntroduceGrid section={story}/>
      <IntroduceGrid section={contents}/>
      <IntroduceGrid section={ai}/>
      <IntroduceGrid section={tensor}/>
      <IntroduceGrid section={models}/>
      <IntroduceGrid section={maths}/>
      <Flex align='center' h='80px'>
        <Divider variant='thick'/>
      </Flex>
      <FootLinks 
        l={footer.l} 
        ll={footer.ll} 
        r={footer.r} 
        rl={footer.rl}
      />
    </Container>
  )
} 

export default Introduction