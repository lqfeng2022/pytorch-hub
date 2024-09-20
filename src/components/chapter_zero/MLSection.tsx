import { Box, Text } from '@chakra-ui/react';
import BaseGrid from '../BaseGrid';
import Definition from '../Definition';
import LeftGrid from '../LeftGrid';
import RightGrid from '../RightGrid';
import chapterZero from '../../data/chapterZero';

const MLSection = () => {
  const models = {
    title: "2.2 Models in Machine Learning",
    value: "A machine learning model is basically a mathematical model that can make predictions or classifications on new data after it's been TRAINED on a dataset. There are many different types of models used in machine learning. Here, I'm gonna briefly introduce three of them for reference."
  }
  const [whats, compare, svms, detree, anns] = chapterZero[1].sections

  return (
    <Box pt={5}>
      <Definition title={whats.name} definition={whats}/>
      <BaseGrid section={compare}/>
      <Box pt={5}>
        <Text as='b' fontSize='lg' color='gray.600'>{models.title}</Text>
        <Text pt={3} fontSize='lg'>{models.value}</Text>
        <BaseGrid section={svms}/>
        <RightGrid section={detree}/>
        <LeftGrid section={anns}/>
      </Box>
    </Box>
  )
}

export default MLSection