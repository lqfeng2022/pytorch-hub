import { Box, Divider, Flex } from '@chakra-ui/react';
import AISection from './chapter_zero/AISection';
import DLSection from './chapter_zero/DLSection';
import Header from './chapter_zero/Header';
import Libraries from './chapter_zero/Libraries';
import MLSection from './chapter_zero/MLSection';
import NNSection from './chapter_zero/NNSection';
import Prerequisites from './chapter_zero/Prerequisites';
import PTSection from './chapter_zero/PTSection';
import TakeCourse from './chapter_zero/TakeCourse';
import ChapterList from './chapter_zero/ChapterList';

const ChapterGrid = () => {
  return (
    <Box maxW='1024px'>
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
    </Box>
  )
}

export default ChapterGrid