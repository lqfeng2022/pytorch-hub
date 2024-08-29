import { Container, Divider, Flex } from '@chakra-ui/react'
import BaseGrid from '../components/BaseGrid'
import ChapterList from '../components/ChapterList'
import CodeDisplay from '../components/CodeDisplay'
import FootLinks from '../components/FootLinks'
import Header from '../components/Header'
import CreateTensorSection from '../components/chapter_one/CreateTensorSection'
import ReproducibilitySection from '../components/chapter_one/ReproducibilitySection'
import TensorAttributs from '../components/chapter_one/TensorAttributs'
import TensorIndex from '../components/chapter_one/TensorIndex'
import TensorManipulate from '../components/chapter_one/TensorManipulate'
import TensorOperations from '../components/chapter_one/TensorOperations'
import TensorRun from '../components/chapter_one/TensorRun'
import TensorSection from '../components/chapter_one/TensorSection'
import sectionList from '../data/sectionList'
import chapterOne from '../data/chapterOne'
import covers from '../data/covers'
import pythonCode from '../data/pythonCode'

const ChapterOne = () => {
  const cover = covers[1]

  const description = sectionList[2].description!
  const lists = sectionList[2].items!
  const leftItems = lists.slice(0, 5)
  const rightItems = lists.slice(-5)

  const [ tanspose ] = chapterOne[5].sections
  const [tensor_numpy] = chapterOne[7].sections

  const tensorCode = pythonCode[0].code
  const attributesCode = pythonCode[5].code

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
      <CodeDisplay codes={tensorCode}/>
      <CreateTensorSection/>
      <TensorAttributs/>
      <CodeDisplay codes={attributesCode}/>
      <TensorOperations/>
      <TensorManipulate/>
      <BaseGrid section={tanspose}/>
      <TensorIndex/>
      <BaseGrid section={tensor_numpy}/>
      <ReproducibilitySection/>
      <TensorRun/>
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