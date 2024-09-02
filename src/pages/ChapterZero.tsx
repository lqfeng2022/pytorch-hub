import { Container, Divider, Flex } from '@chakra-ui/react';
import AISection from '../components/chapter_zero/AISection';
import DLSection from '../components/chapter_zero/DLSection';
import Libraries from '../components/chapter_zero/Libraries';
import MLSection from '../components/chapter_zero/MLSection';
import NNSection from '../components/chapter_zero/NNSection';
import Prerequisites from '../components/chapter_zero/Prerequisites';
import PTSection from '../components/chapter_zero/PTSection';
import FootLinks from '../components/FootLinks';
import covers from '../data/covers';
import Header from '../components/Header';
import sectionList from '../data/sectionList';
import ChapterList from '../components/ChapterList';
import chapterZero from '../data/chapterZero';
import BaseGrid from '../components/BaseGrid';

const ChapterZero = () => {
  const cover = covers[0]

  const descrip = sectionList[1].description!
  const lists = sectionList[1].items!
  const items = lists.slice(0, 8)
  
  const [ takeCourse ] = chapterZero[7].sections

  const footer = {
    l: "Introduction",
    r: "0. TENSORs",
    ll: "/introduction",
    rl: "/tensors"
  }

  return (
    <Container maxW='1200px' px='0'>
      <Header cover={cover}/>
      <Flex align='center' justifyContent='center' h='80px'>
        <Divider variant='brand' w="30%" mx="auto"/>
      </Flex>
      <ChapterList 
        items={items} 
        descrip={descrip} 
      />
      <AISection/>
      <MLSection/>
      <DLSection/>
      <NNSection/>
      <Libraries/>
      <PTSection/>
      <Prerequisites/>
      <BaseGrid section={takeCourse}/>
      <Flex align='center' justifyContent='center' h='80px'>
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

export default ChapterZero