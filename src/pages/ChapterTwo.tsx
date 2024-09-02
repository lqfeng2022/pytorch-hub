import { Container, Divider, Flex } from '@chakra-ui/react'
import ChapterList from '../components/ChapterList'
import Header from '../components/Header'
import covers from '../data/covers'
import sectionList from '../data/sectionList'
import Workflow from '../components/chapter_two/Workflow'
import PrepareData from '../components/chapter_two/PrepareData'
import BuildModel from '../components/chapter_two/BuildModel'
import TrainModel from '../components/chapter_two/TrainModel'
import SaveModel from '../components/chapter_two/SaveModel'

const ChapterTwo = () => {
  const cover = covers[2]

  const descrip = sectionList[3].description!
  const lists = sectionList[3].items!
  const items = lists.slice(0, 5)

  return (
    <Container maxW='1200px' px='10px'>
      <Header cover={cover}/>
      <Flex align='center' h='60px'>
        <Divider variant='brand' w="30%"/>
      </Flex>
      <ChapterList items={items} descrip={descrip}/>
      <Workflow/>
      <PrepareData/>
      <BuildModel/>
      <TrainModel/>
      <SaveModel/>
    </Container>
  )
}

export default ChapterTwo