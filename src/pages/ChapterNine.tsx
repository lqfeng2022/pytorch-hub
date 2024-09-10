import { Container, Divider, Flex } from '@chakra-ui/react'
import FootLinks from '../components/FootLinks'
import Header from '../components/Header'
import covers from '../data/covers'
import sectionList from '../data/sectionList'
import ChapterListTwo from '../components/ChapterListTwo'

const ChapterNine = () => {
  const cover = covers[9]
  const { name: l, link: ll } = sectionList[9]
  const { name: r, link: rl } = sectionList[11]
  const { description: descript, items: lists } = sectionList[10]
  const leftItems = lists.slice(0, 7)
  const rightItems = lists.slice(-6)
  
  return (
    <Container maxW='1200px' px='10px'>
      <Header cover={cover}/>
      <Flex align='center' h='60px'>
        <Divider variant='brand' w='30%' mx='auto'/>
      </Flex>
      <ChapterListTwo 
        leftItems={leftItems} 
        rightItems={rightItems} 
        descrip={descript}
      />
      <div>Chapter Nine comming soon..</div>
      <Flex align='center' h='80px'>
        <Divider variant='thick'/>
      </Flex>
      <FootLinks l={l} ll={ll} r={r} rl={rl}/>
    </Container>
  )
}

export default ChapterNine