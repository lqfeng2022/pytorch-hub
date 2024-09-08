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
import useScrollToHash from '../components/useScrollToHash'
import covers from '../data/covers'
import sectionList from '../data/sectionList'
import '../index.css'

const ChapterOne = () => {
  const cover = covers[1]
  const { description: descript, items: lists } = sectionList[2];
  const [ tensor, create, attrib, oper, manipul, index, reproduc, run 
  ] = lists.slice(0, 8)
  const { name: l, link: ll } = sectionList[1]
  const { name: r, link: rl } = sectionList[3]

  useScrollToHash()

  return (
    <Container maxW='1200px' px='0'>
      <Header cover={cover}/>
      <Flex align='center' h='60px'>
        <Divider variant='brand' w="30%"/>
      </Flex>
      <ChapterList items={lists} descrip={descript}/>
      <div id={tensor.link} className='pt60'><TensorWhats/></div>
      <div id={create.link} className='pt60'><TensorCreate/></div>
      <div id={attrib.link} className='pt60'><TensorAttributs/></div>
      <div id={oper.link} className='pt60'><TensorOperations/></div>
      <div id={manipul.link} className='pt60'><TensorManipulate/></div>
      <div id={index.link} className='pt60'><TensorIndex/></div>
      <div id={reproduc.link} className='pt60'><TensorReproducibility/></div>
      <div id={run.link} className='pt60'><TensorRun/></div>
      <Flex align='center' h='80px'>
        <Divider variant='thick'/>
      </Flex>
      <FootLinks l={l} ll={ll} r={r} rl={rl}/>
    </Container>
  )
}

export default ChapterOne