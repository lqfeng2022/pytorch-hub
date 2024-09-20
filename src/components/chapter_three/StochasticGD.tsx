import chapterThree from '../../data/chapterThree'
import Definition from '../Definition'
import BaseGrid from '../BaseGrid'
import { Box } from '@chakra-ui/react'

const StochasticGD = () => {
  const [ whats, sgd_one, sgd_two ] = chapterThree[4].sections

  return (
    <Box pt={5}>
      <Definition title={whats.name} definition={whats}/>
      <BaseGrid section={sgd_one}/>
      <BaseGrid section={sgd_two}/>
    </Box>
  )
}

export default StochasticGD