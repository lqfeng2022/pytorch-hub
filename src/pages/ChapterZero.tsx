import { Container, Divider, Flex } from '@chakra-ui/react';
import { useEffect } from 'react';
import { useLocation } from 'react-router-dom';
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
import chapterZero from '../data/chapterZero';
import covers from '../data/covers';
import sectionList from '../data/sectionList';

const ChapterZero = () => {
  const cover = covers[0]
  const { description: descript, items: lists } = sectionList[1];
  const [ takeCourse ] = chapterZero[7].sections
  const { name: l, link: ll } = sectionList[0]
  const { name: r, link: rl } = sectionList[2]

  // Anchor links setting (<a href="#section1">)
  const pt60 = { paddingTop: '60px' }
  const location = useLocation()
  useEffect(() => {
    if (location.hash) {
      const id = location.hash.substring(1) // Remove the '#' from the hash
      const element = document.getElementById(id)
      // Scroll to element and then adjust for the 60px fixed header
      if (element) element.scrollIntoView()
    } 
    else window.scrollTo({ top: 0})
  }, [location]);

  return (
    <Container maxW='1200px' px='0'>
      <Header cover={cover}/>
      <Flex align='center' justifyContent='center' h='80px'>
        <Divider variant='brand' w="30%" mx="auto"/>
      </Flex>
      <ChapterList items={lists} descrip={descript}/>
      <div id={lists[0].link} style={pt60}><AISection/></div>
      <div id={lists[1].link} style={pt60}><MLSection/></div>
      <div id={lists[2].link} style={pt60}><DLSection/></div>
      <div id={lists[3].link} style={pt60}><NNSection/></div>
      <div id={lists[4].link} style={pt60}><Libraries/></div>
      <div id={lists[5].link} style={pt60}><PTSection/></div>
      <div id={lists[6].link} style={pt60}><Prerequisites/></div>
      <div id={lists[7].link} style={pt60}><BaseGrid section={takeCourse}/></div>
      <Flex align='center' justifyContent='center' h='80px'>
        <Divider variant='thick'/>
      </Flex>
      <FootLinks l={l} ll={ll} r={r} rl={rl}/>
    </Container>
  )
}

export default ChapterZero