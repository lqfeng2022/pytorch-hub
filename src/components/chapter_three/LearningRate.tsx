import { Box } from '@chakra-ui/react'
import chapterThree from '../../data/chapterThree'
import BaseGrid from '../BaseGrid'
import Definition from '../Definition'
import LeftGrid from '../LeftGrid'
import RightGrid from '../RightGrid'

const LearningRate = () => {
  const [ 
    whats, lr, lr_low, lr_high, lr_right ] = chapterThree[5].sections

  return (
    <Box pt={5}>
      <Definition title={whats.name} definition={whats}/>
      <BaseGrid section={lr}/>
      <LeftGrid section={lr_low}/>
      <RightGrid section={lr_high}/>
      <LeftGrid section={lr_right}/>
    </Box>
  )
}

export default LearningRate