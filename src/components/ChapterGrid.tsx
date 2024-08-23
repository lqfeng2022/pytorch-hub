import { Box, Divider, Flex, Heading, Text } from '@chakra-ui/react';
import ChapterCover from './ChapterCover';
import ChapterList from './ChapterList';
import ChapterQuote from './ChapterQuote';
import Definition from './Definition';
import WhyAI from './WhyAI';
import HistroyAI from './HistroyAI';

const ChapterGrid = () => {
  const title = {name: "Chapter 0: Artificial Intelligence"}

  return (
    <Box p={3} maxW='800px'>
      <Heading size='xl' textAlign='center' py={3}>{title.name}</Heading>
      <ChapterQuote/>
      <ChapterCover/>
      <Flex align='center' justifyContent='center' h='80px'>
        <Divider variant='brand' w="30%" mx="auto"/>
      </Flex>
      <ChapterList/>
      {/* Section One */}
      <Text as='b' fontSize='xl'>1. Artificial Intelligence</Text>
      <Definition/>
      <WhyAI/>
      <HistroyAI/>
    </Box>
  )
}

export default ChapterGrid