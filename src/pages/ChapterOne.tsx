import { Container, Divider, Flex } from '@chakra-ui/react'
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
import useScrollToHash from '../components/useScrollToHash'
import covers from '../data/covers'
import sectionList from '../data/sectionList'
import ChapterListTwo from '../components/ChapterListTwo'

const ChapterOne = () => {
  const cover = covers[1]
  const { description: descript, items: lists } = sectionList[2];
  const lItems = lists.slice(0, 4)
  const rItems = lists.slice(-4)
  const [ tensor, create, attrib, oper, manipul, index, reproduc, run 
  ] = lists.slice(0, 8)
  const { name: l, link: ll } = sectionList[1]
  const { name: r, link: rl } = sectionList[3]

  useScrollToHash()

  return (
    <Container maxW='1200px' px='0'>
      <Header cover={cover}/>
      <Flex align='center' h='60px'>
        <Divider variant='brand' w='30%' mx='auto'/>
      </Flex>
      <ChapterListTwo leftItems={lItems} rightItems={rItems} descrip={descript}/>
      <div id={tensor.link}><TensorWhats/></div>
      <div id={create.link}><TensorCreate/></div>
      <div id={attrib.link}><TensorAttributs/></div>
      <div id={oper.link}><TensorOperations/></div>
      <div id={manipul.link}><TensorManipulate/></div>
      <div id={index.link}><TensorIndex/></div>
      <div id={reproduc.link}><TensorReproducibility/></div>
      <div id={run.link}><TensorRun/></div>
      <Flex align='center' h='80px'>
        <Divider variant='thick'/>
      </Flex>
      <FootLinks l={l} ll={ll} r={r} rl={rl}/>
    </Container>
  )
}

export default ChapterOne