import { Container, Divider, Flex } from '@chakra-ui/react'
import ChapterList from '../components/ChapterList'
import FootLinks from '../components/FootLinks'
import Header from '../components/Header'
import {
  BuildModel,
  PrepareData,
  SaveModel,
  TrainModel,
  Workflow
} from '../components/chapter_two'
import useScrollToHash from '../components/useScrollToHash'
import covers from '../data/covers'
import sectionList from '../data/sectionList'
import '../index.css'

const ChapterTwo = () => {
  const cover = covers[2]
  const { description: descript, items: lists } = sectionList[3];
  const [ flow, prepare, build, train, save ] = lists.slice(0, 7)
  const { name: l, link: ll } = sectionList[2]
  const { name: r, link: rl } = sectionList[4]

  useScrollToHash()

  return (
    <Container maxW='1200px' px='10px'>
      <Header cover={cover}/>
      <Flex align='center' h='60px'>
        <Divider variant='brand' w='30%' ml='auto'/>
      </Flex>
      <ChapterList items={lists} descrip={descript}/>
      <div id={flow.link} className='pt60'><Workflow/></div>
      <div id={prepare.link} className='pt60'><PrepareData/></div>
      <div id={build.link} className='pt60'><BuildModel/></div>
      <div id={train.link} className='pt60'><TrainModel/></div>
      <div id={save.link} className='pt60'><SaveModel/></div>
      <Flex align='center' h='80px'>
        <Divider variant='thick'/>
      </Flex>
      <FootLinks l={l} ll={ll} r={r} rl={rl}/>
    </Container>
  )
}

export default ChapterTwo