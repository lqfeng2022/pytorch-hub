import { Box, Divider, Flex } from '@chakra-ui/react';
import ChapterCover from './ChapterCover';
import ChapterList from './ChapterList';
import ChapterQuote from './ChapterQuote';
import ChapterTitle from './ChapterTitle';

const ChapterGrid = () => {
  return (
    <Box p={3} maxW='800px'>
      <ChapterTitle/>
      <ChapterQuote/>
      <ChapterCover/>
      <Flex align='center' justifyContent='center' h='50px'>
          <Divider variant='brand' w="30%" mx="auto"/>
      </Flex>
      <ChapterList/>
    </Box>
  )
}

export default ChapterGrid