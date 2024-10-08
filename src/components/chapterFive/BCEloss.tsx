import { Box } from '@chakra-ui/react'
import chapterFive from '../../data/chapterFive'
import BaseGrid from '../BaseGrid'
import Definition from '../Definition'

const BCEloss = () => {
  const [ 
    defin, bceFormula, crossEntropy, entropy ]= chapterFive[1].sections

  return (
    <Box pt={5}>
      <Definition title={defin.name} definition={defin} />
      <BaseGrid section={bceFormula}/>
      <Definition title={crossEntropy.name} definition={crossEntropy}/>
      <Definition title={entropy.name} definition={entropy}/>
    </Box>
  )
}

export default BCEloss