import { Container, Divider, Flex } from '@chakra-ui/react'
import covers from '../data/covers'
import Header from '../components/Header'
import chapterList from '../data/chapterList'
import ChapterList from '../components/ChapterList'
import FootLinks from '../components/FootLinks'
import TensorSection from '../components/chapter_one/TensorSection'
import CreateTensorSection from '../components/chapter_one/CreateTensorSection'
import TensorAttributs from '../components/chapter_one/TensorAttributs'
import TensorOperations from '../components/chapter_one/TensorOperations'
import TensorManipulate from '../components/chapter_one/TensorManipulate'
import BaseGrid from '../components/BaseGrid'
import chapterOne from '../data/chapterOne'

const ChapterOne = () => {
  const cover = covers[1].cover

  const description = chapterList[2].description!
  const lists = chapterList[2].items!
  const leftItems = lists.slice(0, 5)
  const rightItems = lists.slice(-5)

  const [ tanspose ] = chapterOne[5].sections

  return (
    <Container maxW='1200px' px='0'>
      <Header cover={cover}/>
      <Flex align='center' justifyContent='center' h='80px'>
        <Divider variant='brand' w="30%" mx="auto"/>
      </Flex>
      <ChapterList 
        leftItems={leftItems} 
        rightItems={rightItems} 
        description={description} 
      />
      <TensorSection/>
      <CreateTensorSection/>
      <TensorAttributs/>
      <TensorOperations/>
      <TensorManipulate/>
      <BaseGrid section={tanspose}/>
      <Flex align='center' justifyContent='center' h='80px'>
        <Divider variant='thick'/>
      </Flex>
      <FootLinks 
        left={'0. Artificial Intelligence'}
        leftLink={'/artificial-intelligence'} 
        right={'1. A Straight Line Model'}
        rightLink={'/a-straight-line-model'}
      />
    </Container>
  )
}

export default ChapterOne