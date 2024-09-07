import { Container, Divider, Flex } from '@chakra-ui/react'
import ChapterList from '../components/ChapterList'
import FootLinks from '../components/FootLinks'
import Header from '../components/Header'
import {
  BuildModel,
  PrepareData,
  SaveModel,
  TrainModel,
  Workflow
} from '../components/chapter_two'
import covers from '../data/covers'
import sectionList from '../data/sectionList'

const ChapterTwo = () => {
  const cover = covers[2]
  const { description: descript, items: lists } = sectionList[3];
  const items = lists.slice(0, 7)
  const { name: l, link: ll } = sectionList[2]
  const { name: r, link: rl } = sectionList[4]

  return (
    <Container maxW='1200px' px='10px'>
      <Header cover={cover}/>
      <Flex align='center' h='60px'>
        <Divider variant='brand' w="30%"/>
      </Flex>
      <ChapterList items={items} descrip={descript}/>
      <Workflow/>
      <PrepareData/>
      <BuildModel/>
      <TrainModel/>
      <SaveModel/>
      <Flex align='center' h='80px'>
        <Divider variant='thick'/>
      </Flex>
      <FootLinks l={l} ll={ll} r={r} rl={rl}/>
    </Container>
  )
}

export default ChapterTwo