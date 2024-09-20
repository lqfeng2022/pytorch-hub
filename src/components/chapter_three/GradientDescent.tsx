import chapterThree from '../../data/chapterThree'
import Definition from '../Definition'
import LeftGrid from '../LeftGrid'
import BaseGrid from '../BaseGrid'
import { Box } from '@chakra-ui/react'

const GradientDescent = () => {
  const [ whats, gdone, gdone_table, gdone_mse, gdtwo_visual, gdtwo_table, gdtwo 
  ] = chapterThree[3].sections

  return (
    <Box pt={5}>  
      <Definition title={whats.name} definition={whats}/>
      <BaseGrid section={gdone}/>
      <LeftGrid section={gdone_table}/>
      <BaseGrid section={gdone_mse}/>
      <BaseGrid section={gdtwo_visual}/>
      <BaseGrid section={gdtwo_table}/>
      <BaseGrid section={gdtwo}/>
    </Box>
  )
}

export default GradientDescent