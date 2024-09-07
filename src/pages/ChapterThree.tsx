import { Container, Divider, Flex } from '@chakra-ui/react'
import ChapterList from '../components/ChapterList'
import FootLinks from '../components/FootLinks'
import Header from '../components/Header'
import covers from '../data/covers'
import sectionList from '../data/sectionList'
import {
  GradientDescent,
  LearningRate,
  LinearRegression,
  LossCurves,
  NormalDistribution,
  StochasticGD
} from '../components/chapter_three'

const ChapterThree = () => {
  const cover = covers[3]
  const { description: descript, items: lists } = sectionList[4];
  const items = lists.slice(0, 6)
  const { name: l, link: ll } = sectionList[3]
  const { name: r, link: rl } = sectionList[5]

  return (
    <Container maxW='1200px' px='10px'>
      <Header cover={cover}/>
      <Flex align='center' h='60px'>
        <Divider variant='brand' w="30%"/>
      </Flex>
      <ChapterList items={items} descrip={descript}/>
      <Flex align='center' h='80px'>
        <Divider variant='thick'/>
      </Flex>
      <LinearRegression/>
      <NormalDistribution/>
      <LossCurves/>
      <GradientDescent/>
      <StochasticGD/>
      <LearningRate/>
      <Flex align='center' h='80px'>
        <Divider variant='thick'/>
      </Flex>
      <FootLinks l={l} ll={ll} r={r} rl={rl}/>
    </Container>
  )
}

export default ChapterThree