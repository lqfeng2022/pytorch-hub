import { Box } from '@chakra-ui/react'
import chapterFive from '../../data/chapterFive'
import BaseGrid from '../BaseGrid'
import Definition from '../Definition'
import LeftGrid from '../LeftGrid'

const SigmoidFunction = () => {
  const [ defin, formula, features ] = chapterFive[2].sections

  return (
    <Box pt={5}>
      <Definition title={defin.name} definition={defin}/>
      <LeftGrid section={formula}/>
      <BaseGrid section={features}/>
    </Box>
  )
}

export default SigmoidFunction