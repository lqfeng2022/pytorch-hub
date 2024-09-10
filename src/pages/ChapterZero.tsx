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
import FootLinks from '../components/FootLinks';
import Header from '../components/Header';
import useScrollToHash from '../components/useScrollToHash';
import chapterZero from '../data/chapterZero';
import covers from '../data/covers';
import sectionList from '../data/sectionList';
import ChapterListTwo from '../components/ChapterListTwo';

const ChapterZero = () => {
  const cover = covers[0]
  const { description: descript, items: lists } = sectionList[1]
  const lItems = lists.slice(0, 4)
  const rItems = lists.slice(-4)
  const [ ai, ml, dl, nn, lib, pt, pre, take ] = lists.slice(0, 8)
  const { name: l, link: ll } = sectionList[0]
  const { name: r, link: rl } = sectionList[2]
  const [ takeCourse ] = chapterZero[7].sections

  useScrollToHash() // Anchor link navigation

  return (
    <Container maxW='1200px' px='0'>
      <Header cover={cover}/>
      <Flex align='center' justifyContent='center' h='80px'>
        <Divider variant='brand' w='30%'/>
      </Flex>
      <ChapterListTwo leftItems={lItems} rightItems={rItems} descrip={descript}/>
      <div id={ai.link}><AISection/></div>
      <div id={ml.link}><MLSection/></div>
      <div id={dl.link}><DLSection/></div>
      <div id={nn.link}><NNSection/></div>
      <div id={lib.link}><Libraries/></div>
      <div id={pt.link}><PTSection/></div>
      <div id={pre.link}><Prerequisites/></div>
      <div id={take.link}><BaseGrid section={takeCourse}/></div>
      <Flex align='center' justifyContent='center' h='80px'>
        <Divider variant='thick'/>
      </Flex>
      <FootLinks l={l} ll={ll} r={r} rl={rl}/>
    </Container>
  )
}

export default ChapterZero