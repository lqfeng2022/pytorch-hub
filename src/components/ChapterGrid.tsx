import { Box } from '@chakra-ui/react';
import AISection from './chapter_zero/AISection';
import DLSection from './chapter_zero/DLSection';
import Header from './chapter_zero/Header';
import Libraries from './chapter_zero/Libraries';
import MLSection from './chapter_zero/MLSection';
import NNSection from './chapter_zero/NNSection';
import Prerequisites from './chapter_zero/Prerequisites';
import PTSection from './chapter_zero/PTSection';
import TakeCourse from './chapter_zero/TakeCourse';

const ChapterGrid = () => {
  return (
    <Box p={3} maxW='900px'>
      <Header/>
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