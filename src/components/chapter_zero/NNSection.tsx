import BaseGrid from '../BaseGrid'
import Definition from '../Definition'
import chapterZero from '../../data/chapterZero'
import { Box } from '@chakra-ui/react'

const NNSection = () => {
  const [ whats, model, neurons ] = chapterZero[3].sections
  
  return (
    <Box pt={5}>
      <Definition title={whats.name} definition={whats}/>
      <BaseGrid section={model}/>
      <BaseGrid section={neurons}/>
    </Box>
  )
}

export default NNSection