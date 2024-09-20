import chapterFive from '../../data/chapterFive'
import Definition from '../Definition'
import BaseGrid from '../BaseGrid'
import { Box } from '@chakra-ui/react'

const Backpropagation = () => {
  const [ defin, implem, calcul ] = chapterFive[3].sections

  return (
    <Box pt={5}>
      <Definition title={defin.name} definition={defin}/>
      <BaseGrid section={implem}/>
      <BaseGrid section={calcul}/>
    </Box>
  )
}

export default Backpropagation