import { Container, Divider, Flex } from '@chakra-ui/react'
import ChapterList from '../components/ChapterList'
import Header from '../components/Header'
import covers from '../data/covers'
import sectionList from '../data/sectionList'
import FootLinks from '../components/FootLinks'
import LinearRegression from '../components/chapter_three/LinearRegression'
import NormalDistribution from '../components/chapter_three/NormalDistribution'
import LossCurves from '../components/chapter_three/LossCurves'
import GradientDescent from '../components/chapter_three/GradientDescent'
import StochasticGD from '../components/chapter_three/StochasticGD'
import LearningRate from '../components/chapter_three/LearningRate'

const ChapterThree = () => {
  const cover = covers[3]

  const descript = sectionList[4].description!
  const lists = sectionList[4].items!
  const items = lists.slice(0, 6)

  const footer = {
    l: "2. A Straight Line Model",
    r: "4. A Binary Classification Model",
    ll: "/a-straight-line-model",
    rl: "/a-binary-classification-model"
  }

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
      <FootLinks 
        l={footer.l} 
        ll={footer.ll} 
        r={footer.r} 
        rl={footer.rl}
      />
    </Container>
  )
}

export default ChapterThree