import { Container, Divider, Flex } from '@chakra-ui/react'
import Header from '../components/Header'
import ChapterList from '../components/ChapterList'
import FootLinks from '../components/FootLinks'
import covers from '../data/covers'
import sectionList from '../data/sectionList'
import Classification from '../components/chapterFive/Classification'
import BCEloss from '../components/chapterFive/BCEloss'
import Sigmoid from '../components/chapterFive/SigmoidFunction'
import Backpropagation from '../components/chapterFive/Backpropagation'
import ReLUfunction from '../components/chapterFive/ReLUfunction'

const ChapterFive = () => {
  const cover = covers[5]
  const descript = sectionList[6].description!
  const lists = sectionList[6].items!
  const items = lists.slice(0, 5)

  const footer = {
    l: "4. A Binary Classification Model",
    r: "6. A CNN Model",
    ll: "/a-binary-classification-model",
    rl: "/a-cnn-model"
  }
  
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
        <Sigmoid/>
        <ReLUfunction/>
        <Backpropagation/>
        <Flex align='center' h='80px'>
          <Divider variant='thick'/>
        </Flex>
        <FootLinks 
          l={footer.l} 
          ll={footer.ll} 
          r={footer.r} 
          rl={footer.rl}
        />
      </div>
    </Container>
  )
}

export default ChapterFive