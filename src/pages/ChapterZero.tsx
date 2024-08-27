import { Container, Divider, Flex } from '@chakra-ui/react';
import AISection from '../components/chapter_zero/AISection';
import DLSection from '../components/chapter_zero/DLSection';
import Header from '../components/chapter_zero/Header';
import Libraries from '../components/chapter_zero/Libraries';
import MLSection from '../components/chapter_zero/MLSection';
import NNSection from '../components/chapter_zero/NNSection';
import Prerequisites from '../components/chapter_zero/Prerequisites';
import PTSection from '../components/chapter_zero/PTSection';
import TakeCourse from '../components/chapter_zero/TakeCourse';
import ChapterList from '../components/chapter_zero/ChapterList';

const ChapterZero = () => {
  return (
    <Container maxW='1200px' px='10px'>
      <Header/>
      <Flex align='center' justifyContent='center' h='80px'>
        <Divider variant='brand' w="30%" mx="auto"/>
      </Flex>
      <ChapterList/>
      <AISection/>
      <MLSection/>
      <DLSection/>
      <NNSection/>
      <Libraries/>
      <PTSection/>
      <Prerequisites/>
      <TakeCourse/>
    </Container>
  )
}

export default ChapterZero