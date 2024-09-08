import { Container, Divider, Flex, Heading, Text } from '@chakra-ui/react'
import FootLinks from '../components/FootLinks'
import IntroduceGrid from '../components/IntroduceGrid'
import introduce from '../data/introduce'
import sectionList from '../data/sectionList'

const Introduction = () => {
  const [ 
    intro, story, contents, ai, tensor, models, maths ] = introduce.slice(0, 7)
  const { name: r, link: rl } = sectionList[1]

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
      <FootLinks l={'Book Cover'} ll={'/'} r={r} rl={rl}/>
    </Container>
  )
} 

export default Introduction