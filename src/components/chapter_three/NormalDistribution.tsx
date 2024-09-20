import Definition from '../Definition'
import chapterThree from '../../data/chapterThree'
import BaseGrid from '../BaseGrid'
import { Box } from '@chakra-ui/react'

const NormalDistribution = () => {
  const [ whats, pdf, cdf ]= chapterThree[1].sections

  return (
    <Box pt={5}>
      <Definition title={whats.name} definition={whats}/>
      <BaseGrid section={pdf}/>
      <BaseGrid section={cdf}/>
    </Box>
  )
}

export default NormalDistribution