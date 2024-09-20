import { Container, Divider, Flex } from '@chakra-ui/react'
import { 
  BuildModel, 
  ImproveModel, 
  PrepareData, 
  SaveModel, 
  TrainModel
} from '../components/chapterFour'
import FootLinks from '../components/FootLinks'
import Header from '../components/Header'
import useScrollToHash from '../components/useScrollToHash'
import covers from '../data/covers'
import sectionList from '../data/sectionList'
import ChapterListTwo from '../components/ChapterListTwo'

const ChapterFour = () => {
  const cover = covers[4]

  const { description: descript, items: lists } = sectionList[5];
  const lItems = lists.slice(0, 4)
  const rItems = lists.slice(-4)
  const [ pre, build, train, improve, save ] = lists.slice(0, 5)

  const { name: l, link: ll } = sectionList[4]
  const { name: r, link: rl } = sectionList[6]

  useScrollToHash()
  
  return (
    <Container maxW='1200px' px='10px'>
      <Header cover={cover}/>
      <Flex align='center' h='60px'>
        <Divider variant='brand' w='30%' mx='auto'/>
      </Flex>
      <ChapterListTwo leftItems={lItems} rightItems={rItems} descrip={descript}/>
      <div id={pre.link}><PrepareData/></div>
      <div id={build.link}><BuildModel/></div>
      <div id={train.link}><TrainModel/></div>
      <div id={improve.link}><ImproveModel/></div>
      <div id={save.link}><SaveModel/></div>
      <Flex align='center' h='80px'>
        <Divider variant='thick'/>
      </Flex>
      <FootLinks l={l} ll={ll} r={r} rl={rl}/>
    </Container>
  )
}

export default ChapterFour