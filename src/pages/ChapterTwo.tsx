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
import FootLinks from '../components/FootLinks'

const ChapterTwo = () => {
  const cover = covers[2]

  const descript = sectionList[3].description!
  const lists = sectionList[3].items!
  const items = lists.slice(0, 7)

  const footer = {
    l: "1. TENSORs",
    r: "3. The Maths Behind (I)",
    ll: "/tensors",
    rl: "/the-maths-behind-one"
  }

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
      <FootLinks 
        l={footer.l} 
        ll={footer.ll} 
        r={footer.r} 
        rl={footer.rl}
      />
    </Container>
  )
}

export default ChapterTwo