import { Container, Divider, Flex } from '@chakra-ui/react'
import sectionList from '../data/sectionList'
import FootLinks from '../components/FootLinks'
import covers from '../data/covers'
import Header from '../components/Header'
import ChapterListTwo from '../components/ChapterListTwo'

const ChapterSeven = () => {
  const cover = covers[7]
  const { description: descript, items: lists } = sectionList[8]
  const lItems = lists.slice(0, 4)
  const rItems = lists.slice(-4)
  const { name: l, link: ll } = sectionList[7]
  const { name: r, link: rl } = sectionList[9]
  
  return (
    <Container maxW='1200px' px='10px'>
      <Header cover={cover}/>
      <Flex align='center' h='60px'>
        <Divider variant='brand' w='30%' mx='auto'/>
      </Flex>
      <ChapterListTwo leftItems={lItems} rightItems={rItems} descrip={descript}/>
      <div>Chapter Seven comming soon..</div>
      <Flex align='center' h='80px'>
        <Divider variant='thick'/>
      </Flex>
      <FootLinks l={l} ll={ll} r={r} rl={rl}/>
    </Container>
  )
}

export default ChapterSeven