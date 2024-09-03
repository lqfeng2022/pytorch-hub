import { Container, Divider, Flex } from '@chakra-ui/react'
import Header from '../components/Header'
import ChapterList from '../components/ChapterList'
import FootLinks from '../components/FootLinks'
import covers from '../data/covers'
import sectionList from '../data/sectionList'
import PrepareData from '../components/chapterFour/PrepareData'
import BuildModel from '../components/chapterFour/BuildModel'
import TrainModel from '../components/chapterFour/TrainModel'
import ImproveModel from '../components/chapterFour/ImproveModel'
import SaveModel from '../components/chapterFour/SaveModel'

const ChapterFour = () => {
  const cover = covers[4]
  const descript = sectionList[5].description!
  const lists = sectionList[5].items!
  const items = lists.slice(0, 5)

  const footer = {
    l: "3. The Maths Behind (I)",
    r: "5. The Maths Behind (II)",
    ll: "/the-maths-behind-one",
    rl: "/the-maths-behind-two"
  }
  
  return (
    <Container maxW='1200px' px='10px'>
      <Header cover={cover}/>
      <Flex align='center' h='60px'>
        <Divider variant='brand' w="30%"/>
      </Flex>
      <ChapterList items={items} descrip={descript}/>
      <Flex align='center' h='80px'>
        <Divider variant='thick'/>
      </Flex>
      <PrepareData/>
      <BuildModel/>
      <TrainModel/>
      <ImproveModel/>
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

export default ChapterFour