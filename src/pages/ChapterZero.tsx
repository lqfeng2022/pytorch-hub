import { Container, Divider, Flex } from '@chakra-ui/react';
import BaseGrid from '../components/BaseGrid';
import {
  AISection,
  DLSection,
  Libraries,
  MLSection,
  NNSection,
  PTSection,
  Prerequisites
} from '../components/chapter_zero';
import ChapterList from '../components/ChapterList';
import FootLinks from '../components/FootLinks';
import Header from '../components/Header';
import useScrollToHash from '../components/useScrollToHash';
import chapterZero from '../data/chapterZero';
import covers from '../data/covers';
import sectionList from '../data/sectionList';
import '../index.css';

const ChapterZero = () => {
  const cover = covers[0]
  const { description: descript, items: lists } = sectionList[1]
  const [ ai, ml, dl, nn, lib, pt, pre, take ] = lists.slice(0, 8)
  const { name: l, link: ll } = sectionList[0]
  const { name: r, link: rl } = sectionList[2]
  const [ takeCourse ] = chapterZero[7].sections

  useScrollToHash() // Anchor link navigation

  return (
    <Container maxW='1200px' px='0'>
      <Header cover={cover}/>
      <Flex align='center' justifyContent='center' h='80px'>
        <Divider variant='brand' w="30%" mx="auto"/>
      </Flex>
      <ChapterList items={lists} descrip={descript}/>
      <div id={ai.link} className='pt60'><AISection/></div>
      <div id={ml.link} className='pt60'><MLSection/></div>
      <div id={dl.link} className='pt60'><DLSection/></div>
      <div id={nn.link} className='pt60'><NNSection/></div>
      <div id={lib.link} className='pt60'><Libraries/></div>
      <div id={pt.link} className='pt60'><PTSection/></div>
      <div id={pre.link} className='pt60'><Prerequisites/></div>
      <div id={take.link} className='pt60'><BaseGrid section={takeCourse}/></div>
      <Flex align='center' justifyContent='center' h='80px'>
        <Divider variant='thick'/>
      </Flex>
      <FootLinks l={l} ll={ll} r={r} rl={rl}/>
    </Container>
  )
}

export default ChapterZero