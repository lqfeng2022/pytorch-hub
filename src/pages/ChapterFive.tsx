import { Container, Divider, Flex } from '@chakra-ui/react'
import ChapterList from '../components/ChapterList'
import FootLinks from '../components/FootLinks'
import Header from '../components/Header'
import covers from '../data/covers'
import sectionList from '../data/sectionList'
import {
  Backpropagation,
  BCEloss,
  Classification,
  ReLUfunction,
  SigmoidFunction
} from '../components/chapterFive'

const ChapterFive = () => {
  const cover = covers[5]
  const { description: descript, items: lists } = sectionList[6];
  const items = lists.slice(0, 5)
  const { name: l, link: ll } = sectionList[5]
  const { name: r, link: rl } = sectionList[7]

  return (
    <Container maxW='1200px' px='10px'>
      <div>
        <Header cover={cover}/>
        <Flex align='center' h='60px'>
          <Divider variant='brand' w="30%"/>
        </Flex>
        <ChapterList items={items} descrip={descript}/>
        <Classification/>
        <BCEloss/>
        <SigmoidFunction/>
        <ReLUfunction/>
        <Backpropagation/>
        <Flex align='center' h='80px'>
          <Divider variant='thick'/>
        </Flex>
        <FootLinks l={l} ll={ll} r={r} rl={rl}/>
      </div>
    </Container>
  )
}

export default ChapterFive