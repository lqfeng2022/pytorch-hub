import { Container, Divider, Flex } from '@chakra-ui/react'
import sectionList from '../data/sectionList'
import FootLinks from '../components/FootLinks'
import Header from '../components/Header'
import ChapterList from '../components/ChapterList'
import covers from '../data/covers'

const ChapterEight = () => {
  const cover = covers[8]
  const { description: descript, items: lists } = sectionList[9]
  const { name: l, link: ll } = sectionList[8]
  const { name: r, link: rl } = sectionList[10]
  
  return (
    <Container maxW='1200px' px='10px'>
      <Header cover={cover}/>
      <Flex align='center' h='60px'>
        <Divider variant='brand' w="30%"/>
      </Flex>
      <ChapterList items={lists} descrip={descript}/>
      <div>Chapter Eight comming soon..</div>
      <Flex align='center' h='80px'>
        <Divider variant='thick'/>
      </Flex>
      <FootLinks l={l} ll={ll} r={r} rl={rl}/>
    </Container>
  )
}

export default ChapterEight