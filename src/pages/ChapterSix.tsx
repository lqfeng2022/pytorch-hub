import { Container, Divider, Flex } from '@chakra-ui/react'
import FootLinks from '../components/FootLinks'
import sectionList from '../data/sectionList'
import Header from '../components/Header'
import ChapterList from '../components/ChapterList'
import covers from '../data/covers'

const ChapterSix = () => {
  const cover = covers[6]
  const { description: descript, items: lists } = sectionList[7]
  const { name: l, link: ll } = sectionList[6]
  const { name: r, link: rl } = sectionList[8]
  
  return (
    <Container maxW='1200px' px='10px'>
      <Header cover={cover}/>
      <Flex align='center' h='60px'>
        <Divider variant='brand' w='30%'/>
      </Flex>
      <ChapterList items={lists} descrip={descript}/>
      <div>Chapter Six comming soon..</div>
      <Flex align='center' h='80px'>
        <Divider variant='thick'/>
      </Flex>
      <FootLinks l={l} ll={ll} r={r} rl={rl}/>
    </Container>
  )
}

export default ChapterSix