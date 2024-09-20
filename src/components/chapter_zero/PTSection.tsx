import BaseGrid from '../BaseGrid';
import Definition from '../Definition';
import chapterZero from '../../data/chapterZero';
import { Box } from '@chakra-ui/react';

const PTSection = () => {
  const [ whats, features, trends, companies ] = chapterZero[5].sections

  return (
    <Box pt={5}>
      <Definition title={whats.name} definition={whats}/>
      <BaseGrid section={features}/>
      <BaseGrid section={trends}/>
      <BaseGrid section={companies}/>
    </Box>
  )
}

export default PTSection