import { Container, Divider, Flex } from '@chakra-ui/react'
import sectionList from '../data/sectionList'
import FootLinks from '../components/FootLinks'
import covers from '../data/covers'
import ChapterList from '../components/ChapterList'
import Header from '../components/Header'

const ChapterTen = () => {
  const cover = covers[10]
  const { description: descript, items: lists } = sectionList[11]
  const { name: l, link: ll } = sectionList[10]
  const { name: r, link: rl } = sectionList[12]
  
  return (
    <Container maxW='1200px' px='10px'>
      <Header cover={cover}/>
      <Flex align='center' h='60px'>
        <Divider variant='brand' w='30%' ml='auto'/>
      </Flex>
      <ChapterList items={lists} descrip={descript}/>
      <div>Chapter Ten comming soon..</div>
      <Flex align='center' h='80px'>
        <Divider variant='thick'/>
      </Flex>
      <FootLinks l={l} ll={ll} r={r} rl={rl}/>
    </Container>
  )
}

export default ChapterTen