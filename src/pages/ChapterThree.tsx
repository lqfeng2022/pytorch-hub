import { Container, Divider, Flex } from '@chakra-ui/react'
import {
  GradientDescent,
  LearningRate,
  LinearRegression,
  LossCurves,
  NormalDistribution,
  StochasticGD
} from '../components/chapter_three'
import ChapterList from '../components/ChapterList'
import FootLinks from '../components/FootLinks'
import Header from '../components/Header'
import useScrollToHash from '../components/useScrollToHash'
import covers from '../data/covers'
import sectionList from '../data/sectionList'
import '../index.css'

const ChapterThree = () => {
  const cover = covers[3]
  const { description: descript, items: lists } = sectionList[4];
  const [ lin, distrib, loss, descent, stochast, lr ] = lists.slice(0, 6)
  const { name: l, link: ll } = sectionList[3]
  const { name: r, link: rl } = sectionList[5]

  useScrollToHash()

  return (
    <Container maxW='1200px' px='10px'>
      <Header cover={cover}/>
      <Flex align='center' h='60px'>
        <Divider variant='brand' w='30%'/>
      </Flex>
      <ChapterList items={lists} descrip={descript}/>
      <div id={lin.link} className='pt60'><LinearRegression/></div>
      <div id={distrib.link} className='pt60'><NormalDistribution/></div>
      <div id={loss.link} className='pt60'><LossCurves/></div>
      <div id={descent.link} className='pt60'><GradientDescent/></div>
      <div id={stochast.link} className='pt60'><StochasticGD/></div>
      <div id={lr.link} className='pt60'><LearningRate/></div>
      <Flex align='center' h='80px'>
        <Divider variant='thick'/>
      </Flex>
      <FootLinks l={l} ll={ll} r={r} rl={rl}/>
    </Container>
  )
}

export default ChapterThree