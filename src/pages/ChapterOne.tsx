import { Container, Divider, Flex } from '@chakra-ui/react'
import ChapterList from '../components/ChapterList'
import FootLinks from '../components/FootLinks'
import Header from '../components/Header'
import TensorCreate from '../components/chapter_one/TensorCreate'
import TensorReproducibility from '../components/chapter_one/TensorReproducibility'
import TensorAttributs from '../components/chapter_one/TensorAttributs'
import TensorIndex from '../components/chapter_one/TensorIndex'
import TensorManipulate from '../components/chapter_one/TensorManipulate'
import TensorOperations from '../components/chapter_one/TensorOperations'
import TensorRun from '../components/chapter_one/TensorRun'
import sectionList from '../data/sectionList'
import covers from '../data/covers'
import TensorWhats from '../components/chapter_one/TensorWhats'

const ChapterOne = () => {
  const cover = covers[1]

  const descrip = sectionList[2].description!
  const lists = sectionList[2].items!
  const litems = lists.slice(0, 5)
  const ritems = lists.slice(-5)

  const footer = {
    left: "0. Artificial Intelligence",
    leftl: "/artificial-intelligence",
    right: "1. A Straight Line Model",
    rightl: "/a-straight-line-model"
  }

  return (
    <Container maxW='1200px' px='0'>
      <Header cover={cover}/>
      <Flex align='center' justifyContent='center' h='80px'>
        <Divider variant='brand' w="30%" mx="auto"/>
      </Flex>
      <ChapterList 
        leftItems={litems} 
        rightItems={ritems} 
        description={descrip}
      />
      <TensorWhats/>
      <TensorCreate/>
      <TensorAttributs/>
      <TensorOperations/>
      <TensorManipulate/>
      <TensorIndex/>
      <TensorReproducibility/>
      <TensorRun/>
      <Flex align='center' justifyContent='center' h='80px'>
        <Divider variant='thick'/>
      </Flex>
      <FootLinks 
        l={footer.left}
        llink={footer.leftl} 
        r={footer.right}
        rl={footer.rightl}
      />
    </Container>
  )
}

export default ChapterOne