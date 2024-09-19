import { Box, Image, Text } from '@chakra-ui/react';
import { ai_history } from '../../assets/chapter_zero';
import BaseGrid from '../BaseGrid';
import Definition from '../Definition';
import LeftGrid from '../LeftGrid';
import RightGrid from '../RightGrid';
import RightGridTwo from '../RightGridTwo';
import chapterZero from '../../data/chapterZero';

const AISection = () => {
  const [
    whats, reason, turningTest, expertSystem, connectionism, cnn, alphago, openai
  ] =  chapterZero[0].sections

  return (
    <>
      <Definition title={whats.name} definition={whats}/>
      <BaseGrid section={reason}/>
      <Box py={5}>
        <Text as='b' fontSize='lg' color='gray.600'>1.2 The History of AI</Text>
        <Image py={5} src={ai_history}/>
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