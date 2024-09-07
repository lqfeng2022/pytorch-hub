import { Container, Divider, Flex } from '@chakra-ui/react'
import ChapterList from '../components/ChapterList'
import FootLinks from '../components/FootLinks'
import Header from '../components/Header'
import {
  TensorAttributs,
  TensorCreate,
  TensorIndex,
  TensorManipulate,
  TensorOperations,
  TensorReproducibility, TensorRun,
  TensorWhats
} from '../components/chapter_one'
import covers from '../data/covers'
import sectionList from '../data/sectionList'

const ChapterOne = () => {
  const cover = covers[1]
  const { description: descript, items: lists } = sectionList[2];
  const items = lists.slice(0, 8)
  const { name: l, link: ll } = sectionList[1]
  const { name: r, link: rl } = sectionList[3]

  return (
    <Container maxW='1200px' px='0'>
      <Header cover={cover}/>
      <Flex align='center' h='60px'>
        <Divider variant='brand' w="30%"/>
      </Flex>
      <ChapterList items={items} descrip={descript}/>
      <TensorWhats/>
      <TensorCreate/>
      <TensorAttributs/>
      <TensorOperations/>
      <TensorManipulate/>
      <TensorIndex/>
      <TensorReproducibility/>
      <TensorRun/>
      <Flex align='center' h='80px'>
        <Divider variant='thick'/>
      </Flex>
      <FootLinks l={l} ll={ll} r={r} rl={rl}/>
    </Container>
  )
}

export default ChapterOne