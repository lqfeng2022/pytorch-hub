import { Box, Image, Text } from '@chakra-ui/react';
import aihistory from '../../assets/chapter_zero/aihistory.jpeg';
import BaseGrid from '../BaseGrid';
import Definition from '../Definition';
import LeftGrid from '../LeftGrid';
import RightGrid from '../RightGrid';
import RightGridTwo from '../RightGridTwo';
import chapterOne from '../../data/chapterOne';


const AISection = () => {
  const [
    whats, reason, turningTest, expertSystem, connectionism, cnn, alphago, openai
  ] =  chapterOne[0].sections

  return (
    <>
      <Box py={5}>
        <Text as='b' fontSize='xl' color='gray.600'>1. Artificial Intelligence</Text>
        <Definition definition={whats}/>
      </Box>
      <BaseGrid section={reason}/>
      <Box py={5}>
        <Text as='b' fontSize='lg' color='gray.600'>1.2 The History of AI</Text>
        <Image py={5} src={aihistory}/>
        <LeftGrid section={turningTest}/>
        <RightGridTwo section={expertSystem}/>
        <LeftGrid section={connectionism}/>
        <BaseGrid section={cnn}/>
        <LeftGrid section={alphago}/>
        <RightGrid section={openai}/>
      </Box>
    </>
  )
}

export default AISection