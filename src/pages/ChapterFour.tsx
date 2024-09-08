import { Container, Divider, Flex } from '@chakra-ui/react'
import {
  BuildModel, ImproveModel, PrepareData, SaveModel, TrainModel
} from '../components/chapterFour'
import ChapterList from '../components/ChapterList'
import FootLinks from '../components/FootLinks'
import Header from '../components/Header'
import useScrollToHash from '../components/useScrollToHash'
import covers from '../data/covers'
import sectionList from '../data/sectionList'
import '../index.css'

const ChapterFour = () => {
  const cover = covers[4]
  const { description: descript, items: lists } = sectionList[5];
  const [ pre, build, train, improve, save ] = lists.slice(0, 5)
  const { name: l, link: ll } = sectionList[4]
  const { name: r, link: rl } = sectionList[6]

  useScrollToHash()
  
  return (
    <Container maxW='1200px' px='10px'>
      <Header cover={cover}/>
      <Flex align='center' h='60px'>
        <Divider variant='brand' w="30%"/>
      </Flex>
      <ChapterList items={lists} descrip={descript}/>
      <Flex align='center' h='80px'>
        <Divider variant='thick'/>
      </Flex>
      <div id={pre.link} className='pt60'><PrepareData/></div>
      <div id={build.link} className='pt60'><BuildModel/></div>
      <div id={train.link} className='pt60'><TrainModel/></div>
      <div id={improve.link} className='pt60'><ImproveModel/></div>
      <div id={save.link} className='pt60'><SaveModel/></div>
      <Flex align='center' h='80px'>
        <Divider variant='thick'/>
      </Flex>
      <FootLinks l={l} ll={ll} r={r} rl={rl}/>
    </Container>
  )
}

export default ChapterFour